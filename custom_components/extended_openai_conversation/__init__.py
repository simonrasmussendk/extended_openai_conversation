"""The OpenAI Conversation integration."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

import voluptuous as vol
import yaml

from openai import AsyncAzureOpenAI, AsyncOpenAI, AuthenticationError, OpenAIError
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_NAME, CONF_API_KEY, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    TemplateError,
)
from homeassistant.helpers import (
    config_validation as cv,
    entity_registry as er,
    device_registry as dr,
    area_registry as ar,
    intent,
    template,
)
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import ulid

from .const import (
    CONF_API_VERSION,
    CONF_ATTACH_USERNAME,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_CONVERSATION_EXPIRATION_TIME,
    CONF_DOMAIN_KEYWORDS,
    CONF_FUNCTIONS,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_ENABLE_STT,
    CONF_ORGANIZATION,
    CONF_PROMPT,
    CONF_SKIP_AUTHENTICATION,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_USE_TOOLS,
    DEFAULT_ATTACH_USERNAME,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONF_FUNCTIONS,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_CONVERSATION_EXPIRATION_TIME,
    DEFAULT_DOMAIN_KEYWORDS,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_SKIP_AUTHENTICATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_USE_TOOLS,
    DOMAIN,
    EVENT_CONVERSATION_FINISHED,
)
from .conversation_store import ConversationStore
from .exceptions import (
    EntityNotExposed,
    EntityNotFound,
    FunctionLoadFailed,
    FunctionNotFound,
    InvalidFunction,
    ParseArgumentsFailed,
    TokenLengthExceededError,
)
from .helpers import (
    get_function_executor,
    is_azure,
    validate_authentication,
    log_openai_interaction,
    get_domain_entity_attributes,
    async_get_preset_for_model,
    get_param_names,
    get_limits,
    create_openai_client,
    resolve_token_parameters,
    execute_openai_request_with_fallbacks,
    build_sampler_kwargs,
    get_model_config,
    get_custom_parameters,
    distribute_custom_parameters,
)
from .services import async_setup_services

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


# hass.data key for agent.
DATA_AGENT = "agent"
DATA_CONVERSATION_STORE = "conversation_store"


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up OpenAI Conversation."""
    # Create a shared conversation store for all agents
    conversation_store = ConversationStore(DEFAULT_CONVERSATION_EXPIRATION_TIME)
    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN][DATA_CONVERSATION_STORE] = conversation_store
    
    await async_setup_services(hass, config)
    return True


async def _async_options_updated(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Handle options update by reloading the entry."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up OpenAI Conversation from a config entry."""

    try:
        await validate_authentication(
            hass=hass,
            api_key=entry.data[CONF_API_KEY],
            base_url=entry.data.get(CONF_BASE_URL),
            api_version=entry.data.get(CONF_API_VERSION),
            organization=entry.data.get(CONF_ORGANIZATION),
            skip_authentication=entry.data.get(
                CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION
            ),
        )
    except AuthenticationError as err:
        _LOGGER.error("Invalid API key: %s", err)
        return False
    except OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    # Get the shared conversation store
    conversation_store = hass.data[DOMAIN].get(DATA_CONVERSATION_STORE)
    if conversation_store is None:
        # Create one if not exists (should not happen normally)
        conversation_store = ConversationStore(DEFAULT_CONVERSATION_EXPIRATION_TIME)
        hass.data[DOMAIN][DATA_CONVERSATION_STORE] = conversation_store

    agent = OpenAIAgent(hass, entry, conversation_store)

    data = hass.data.setdefault(DOMAIN, {}).setdefault(entry.entry_id, {})
    data[CONF_API_KEY] = entry.data[CONF_API_KEY]
    data[DATA_AGENT] = agent
    # Track whether we actually forwarded the STT platform for this entry
    data.setdefault("stt_loaded", False)
    # Track whether we forwarded the AI Task platform for this entry
    data.setdefault("ai_task_loaded", False)

    conversation.async_set_agent(hass, entry, agent)

    # Listen for options updates to toggle STT without restart
    entry.async_on_unload(entry.add_update_listener(_async_options_updated))

    # Forward STT platform setup if enabled in options
    try:
        config = get_model_config(entry)
        if config["enable_stt"]:
            await hass.config_entries.async_forward_entry_setups(entry, ["stt"])
            data["stt_loaded"] = True
    except Exception as err:  # noqa: BLE001
        _LOGGER.error("Failed to set up STT platform: %s", err)

    # Forward AI Task platform setup (always enabled if platform available)
    try:
        await hass.config_entries.async_forward_entry_setups(entry, ["ai_task"])
        data["ai_task_loaded"] = True
    except Exception as err:  # noqa: BLE001
        _LOGGER.error("Failed to set up AI Task platform: %s", err)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OpenAI."""
    # Unload STT platform only if we had set it up for this entry
    entry_data = hass.data.get(DOMAIN, {}).get(entry.entry_id, {})
    try:
        if entry_data.get("stt_loaded"):
            await hass.config_entries.async_unload_platforms(entry, ["stt"])
            entry_data["stt_loaded"] = False
    except Exception as err:  # noqa: BLE001
        _LOGGER.warning("Failed to unload STT platform cleanly: %s", err)

    # Unload AI Task platform only if we had set it up for this entry
    try:
        if entry_data.get("ai_task_loaded"):
            await hass.config_entries.async_unload_platforms(entry, ["ai_task"])
            entry_data["ai_task_loaded"] = False
    except Exception as err:  # noqa: BLE001
        _LOGGER.warning("Failed to unload AI Task platform cleanly: %s", err)

    hass.data[DOMAIN].pop(entry.entry_id)
    conversation.async_unset_agent(hass, entry)
    return True


class OpenAIAgent(conversation.AbstractConversationAgent):
    """OpenAI conversation agent."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, conversation_store: ConversationStore) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.conversation_store = conversation_store
        
        # We'll initialize the client later to avoid blocking operations during __init__
        self.client = None
        self._initialize_client_task = hass.async_create_task(self._initialize_client())
        # Cache for deciding which token parameter a given model/deployment accepts
        # Keyed by the configured model string (can be an Azure deployment name)
        self._token_param_cache: dict[str, str] = {}

    async def _initialize_client(self):
        """Initialize the OpenAI client in an executor to prevent blocking I/O."""
        # Use an executor to prevent blocking I/O from distro library during initialization
        def create_client():
            return create_openai_client(
                hass=self.hass,
                api_key=self.entry.data[CONF_API_KEY],
                base_url=self.entry.data.get(CONF_BASE_URL),
                api_version=self.entry.data.get(CONF_API_VERSION),
                organization=self.entry.data.get(CONF_ORGANIZATION),
            )
        
        # Run the creation in an executor
        self.client = await self.hass.async_add_executor_job(create_client)

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        # Make sure the client is initialized before proceeding
        if self.client is None:
            try:
                await self._initialize_client_task
            except Exception as err:
                _LOGGER.error("Error initializing OpenAI client: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Failed to initialize OpenAI client: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=user_input.conversation_id
                )
                
        exposed_entities = self.get_exposed_entities()

        # Clean expired conversations
        self.conversation_store.clean_expired_conversations()

        # Check if an existing conversation is provided and ensure it's a valid string
        if user_input.conversation_id and isinstance(user_input.conversation_id, str) and user_input.conversation_id.strip():
            # Try to get existing conversation from the store
            messages = self.conversation_store.get_conversation(user_input.conversation_id)
            conversation_id = user_input.conversation_id
        else:
            # Generate a new conversation ID
            conversation_id = ulid.ulid()
            user_input.conversation_id = conversation_id
            messages = None

        # If no existing messages were found (new conversation or expired), create the system message
        if not messages:
            try:
                system_message = self._generate_system_message(
                    exposed_entities, user_input
                )
            except TemplateError as err:
                _LOGGER.error("Error rendering prompt: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem with my template: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )
            messages = [system_message]
            # For new conversations, ensure we have an entry in the conversation store to track sent domains
            self.conversation_store.save_conversation(conversation_id, messages)
        
        user_message = {"role": "user", "content": user_input.text}
        config = get_model_config(self.entry)
        if config["attach_username"]:
            user = user_input.context.user_id
            if user is not None:
                user_message[ATTR_NAME] = user

        messages.append(user_message)
        
        # Check for domain-specific keywords in user input and add relevant attributes
        domain_attributes = await self.check_domain_keywords(user_input)
        if domain_attributes:
            attributes_message = {
                "role": "system",
                "content": f"Additional entity attributes for relevant domains based on keywords in user input(can in some cases be unrelated and ignored):\n```json\n{json.dumps(domain_attributes, default=str)}\n```\n"
            }
            messages.append(attributes_message)
        
        # Save the conversation with user message before processing to ensure sent_domains is updated
        self.conversation_store.save_conversation(conversation_id, messages)

        try:
            query_response = await self.query(user_input, messages, exposed_entities, 0)
        except OpenAIError as err:
            _LOGGER.error(err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I had a problem talking to OpenAI: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )
        except HomeAssistantError as err:
            _LOGGER.error(err, exc_info=err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Something went wrong: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response, conversation_id=conversation_id
            )

        messages.append(query_response.message.model_dump(exclude_none=True))
        
        # Save the updated conversation to the store
        self.conversation_store.save_conversation(conversation_id, messages)

        self.hass.bus.async_fire(
            EVENT_CONVERSATION_FINISHED,
            {
                "response": query_response.response.model_dump(),
                "user_input": user_input,
                "messages": messages,
            },
        )

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(query_response.message.content)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=conversation_id, continue_conversation=self.is_conversation_continued(messages)
        )
    
    def is_conversation_continued(self, messages: list[dict[str, str]]) -> bool:
        """Determine if the conversation should continue based on the last assistant message.

        Returns True if the last message is from the assistant and ends with a question mark.
        """
        if not messages:
            return False

        last_message = messages[-1]
        if not isinstance(last_message, dict):
            return False

        # Only consider messages from the assistant
        if last_message.get("role") != "assistant":
            return False

        content = last_message.get("content")
        if not isinstance(content, str):
            return False

        # Check if the content ends with a question mark (half-width or CJK full-width)
        return content.strip().endswith(("?", "？"))

    def _generate_system_message(
        self, exposed_entities, user_input: conversation.ConversationInput
    ):
        config = get_model_config(self.entry)
        raw_prompt = config["prompt"]
        prompt = self._async_generate_prompt(raw_prompt, exposed_entities, user_input)
        return {"role": "system", "content": prompt}

    def get_exposed_entities(self):
        states = [
            state
            for state in self.hass.states.async_all()
            if async_should_expose(self.hass, conversation.DOMAIN, state.entity_id)
        ]
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        area_registry = ar.async_get(self.hass)
        exposed_entities = []
        
        for state in states:
            entity_id = state.entity_id
            entity = entity_registry.async_get(entity_id)

            aliases = []
            if entity and entity.aliases:
                aliases = entity.aliases
            
            # Get area information
            area_id = None
            area_name = None
            
            # First try to get area from the entity directly
            if entity and entity.area_id:
                area_id = entity.area_id
                area = area_registry.async_get_area(area_id)
                if area:
                    area_name = area.name
            # If no area on entity, try to get it from the device
            elif entity and entity.device_id:
                device = device_registry.async_get(entity.device_id)
                if device and device.area_id:
                    area_id = device.area_id
                    area = area_registry.async_get_area(area_id)
                    if area:
                        area_name = area.name

            exposed_entities.append(
                {
                    "entity_id": entity_id,
                    "name": state.name,
                    "state": self.hass.states.get(entity_id).state,
                    "aliases": aliases,
                    "area_id": area_id,
                    "area_name": area_name,
                }
            )
        return exposed_entities
    
    def get_areas(self):
        """Get all areas from the area registry."""
        area_registry = ar.async_get(self.hass)
        areas = []
        
        for area_id, area in area_registry.areas.items():
            areas.append({
                "area_id": area_id,
                "name": area.name,
                "aliases": area.aliases if hasattr(area, 'aliases') else [],
            })
            
        return areas

    async def check_domain_keywords(self, user_input: conversation.ConversationInput):
        """Check if user input contains keywords for domains and add relevant entity attributes.
        
        Returns a dictionary of domain entity attributes that should be added to the system message.
        
        Expected format for domain_keywords config:
        ```yaml
        - domain: light
          keywords:
            - brightness
            - color
            - dim
        - domain: climate
          keywords:
            - temperature
            - thermostat
            - heat
        ```
        """
        config = get_model_config(self.entry)
        domain_keywords = config["domain_keywords"]
        if not domain_keywords:
            return {}
            
        # Parse domain_keywords from YAML format
        try:
            domain_keywords_dict = yaml.safe_load(domain_keywords)
            if not isinstance(domain_keywords_dict, list):
                _LOGGER.error("Domain keywords configuration is not a list. Please follow the expected format.")
                return {}
        except yaml.YAMLError as err:
            _LOGGER.error("Error parsing domain keywords YAML: %s", err)
            return {}
        except Exception as err:
            _LOGGER.error("Unexpected error parsing domain keywords: %s", err)
            return {}
            
        # Get exposed entities
        exposed_entities = self.get_exposed_entities()
        
        # Get conversation ID and previously sent domains
        conversation_id = user_input.conversation_id
        
        # Ensure we have a valid conversation ID
        if not conversation_id or not isinstance(conversation_id, str) or not conversation_id.strip():
            _LOGGER.warning("Missing or invalid conversation_id when checking domain keywords")
            return {}
            
        # Get sent domains for this conversation
        sent_domains = self.conversation_store.get_sent_domains(conversation_id)
        _LOGGER.debug("Previously sent domains for conversation %s: %s", conversation_id, sent_domains)
        
        # Check user input for keywords and gather attributes for matching domains
        domain_attributes = {}
        text = user_input.text.lower()
        
        for item in domain_keywords_dict:
            # Validate item structure
            if not isinstance(item, dict):
                _LOGGER.warning("Invalid domain keyword item, expected dictionary: %s", item)
                continue
                
            if "domain" not in item:
                _LOGGER.warning("Missing 'domain' key in domain keyword item: %s", item)
                continue
                
            if "keywords" not in item:
                _LOGGER.warning("Missing 'keywords' key in domain keyword item: %s", item)
                continue
                
            domain = item["domain"]
            keywords = item["keywords"]
            
            # Ensure keywords is a list
            if not isinstance(keywords, list):
                _LOGGER.warning("Keywords must be a list for domain %s: %s", domain, keywords)
                continue
                
            # Skip if this domain has already been sent in this conversation
            if domain in sent_domains:
                _LOGGER.debug("Domain %s already sent in this conversation, skipping", domain)
                continue
                
            # Check if any keyword matches
            if any(keyword.lower() in text for keyword in keywords):
                # Get attributes for this domain
                try:
                    domain_attrs = await get_domain_entity_attributes(
                        self.hass,
                        domain,
                        exposed_entities,
                    )
                    
                    if domain_attrs:
                        _LOGGER.debug("Found matching domain %s for keywords in text: %s", domain, text)
                        domain_attributes[domain] = domain_attrs
                        # Mark this domain as sent in this conversation - must come before query is made
                        self.conversation_store.add_sent_domain(conversation_id, domain)
                    else:
                        _LOGGER.debug("No entities found for domain %s", domain)
                except Exception as err:
                    _LOGGER.error("Error getting domain attributes for %s: %s", domain, err)
                    
        return domain_attributes

    def _async_generate_prompt(
        self,
        raw_prompt: str,
        exposed_entities,
        user_input: conversation.ConversationInput,
    ) -> str:
        """Generate a prompt for the user."""
        areas = self.get_areas()
        return template.Template(raw_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
                "exposed_entities": exposed_entities,
                "areas": areas,
                "current_device_id": user_input.device_id,
            },
            parse_result=False,
        )

    def get_functions(self):
        try:
            config = get_model_config(self.entry)
            function = config["functions"]
            result = yaml.safe_load(function) if function else DEFAULT_CONF_FUNCTIONS
            if result:
                for setting in result:
                    function_executor = get_function_executor(
                        setting["function"]["type"]
                    )
                    setting["function"] = function_executor.to_arguments(
                        setting["function"]
                    )
            return result
        except (InvalidFunction, FunctionNotFound) as e:
            raise e
        except:
            raise FunctionLoadFailed()

    async def truncate_message_history(
        self, messages, exposed_entities, user_input: conversation.ConversationInput
    ):
        """Truncate message history."""
        config = get_model_config(self.entry)
        strategy = config["context_truncate_strategy"]

        if strategy == "clear":
            last_user_message_index = None
            for i in reversed(range(len(messages))):
                if messages[i]["role"] == "user":
                    last_user_message_index = i
                    break

            if last_user_message_index is not None:
                del messages[1:last_user_message_index]
                # refresh system prompt when all messages are deleted
                messages[0] = self._generate_system_message(
                    exposed_entities, user_input
                )

    async def query(
        self,
        user_input: conversation.ConversationInput,
        messages,
        exposed_entities,
        n_requests,
    ) -> OpenAIQueryResponse:
        """Process a sentence."""
        config = get_model_config(self.entry)
        model = config["model"]
        max_tokens = config["max_tokens"]
        top_p = config["top_p"]
        temperature = config["temperature"]
        use_tools = config["use_tools"]
        context_threshold = config["context_threshold"]
        functions = list(map(lambda s: s["spec"], self.get_functions()))

        function_call = "auto"
        if n_requests == config["max_function_calls"]:
            function_call = "none"

        tool_kwargs = {"functions": functions, "function_call": function_call}
        if use_tools:
            tool_kwargs = {
                "tools": [{"type": "function", "function": func} for func in functions],
                "tool_choice": function_call,
            }

        if len(functions) == 0:
            tool_kwargs = {}

        _LOGGER.info("Prompt for %s: %s", model, json.dumps(messages))
        
        # Resolve preset for model and derive parameter behavior (non-blocking)
        preset = await async_get_preset_for_model(self.hass, model)
        preset_param_names = get_param_names(preset) if preset else set()
        
        # Resolve token parameters using centralized logic
        token_kwargs, token_param_name = resolve_token_parameters(
            model=model,
            max_tokens=max_tokens,
            preset=preset,
            token_param_cache=self._token_param_cache,
        )
        
        # Log if clamping occurred
        final_max_tokens = token_kwargs[token_param_name]
        if final_max_tokens != max_tokens:
            _LOGGER.debug("Clamped %s from %s to %s based on preset limit", token_param_name, max_tokens, final_max_tokens)
        
        # Build sampler kwargs using centralized logic
        sampler_kwargs = build_sampler_kwargs(
            model=model,
            temperature=temperature,
            top_p=top_p,
            preset=preset,
        )

        # Get custom parameters and distribute them into appropriate categories
        custom_params = get_custom_parameters(self.entry, preset)
        custom_sampler, custom_reasoning, custom_tool, custom_base = distribute_custom_parameters(custom_params)
        
        # Remove any token parameters from custom parameters to avoid conflicts
        custom_sampler = {k: v for k, v in custom_sampler.items() if k not in ['max_tokens', 'max_completion_tokens']}
        custom_base = {k: v for k, v in custom_base.items() if k not in ['max_tokens', 'max_completion_tokens']}
        
        # Merge custom sampler parameters with standard ones
        sampler_kwargs.update(custom_sampler)
        
        # Optional reasoning parameter mapping if preset declares reasoning_effort
        reasoning_kwargs: dict[str, Any] = {}
        if "reasoning_effort" in preset_param_names:
            effort = config["reasoning_effort"]
            if effort:
                reasoning_kwargs = {"reasoning": {"effort": effort}}
        
        # Merge custom reasoning parameters
        reasoning_kwargs.update(custom_reasoning)
        
        # Log custom parameters being used
        if custom_params:
            _LOGGER.debug("Using custom parameters: %s", custom_params)
        
        # Create request data for logging (reflect chosen params)
        request_data = {
            "model": model,
            "messages": messages,
            token_param_name: max_tokens,
            "user": user_input.conversation_id,
            **sampler_kwargs,
            **reasoning_kwargs,
            **tool_kwargs,
        }

        try:
            # Move the OpenAI API call to an executor to prevent blocking I/O operations
            def execute_openai_request():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        execute_openai_request_with_fallbacks(
                            client=self.client,
                            model=model,
                            messages=messages,
                            token_kwargs=token_kwargs,
                            sampler_kwargs=sampler_kwargs,
                            reasoning_kwargs=reasoning_kwargs,
                            tool_kwargs={**tool_kwargs, **custom_tool},
                            base_kwargs={"user": user_input.conversation_id, **custom_base},
                            token_param_cache=self._token_param_cache,
                        )
                    )
                finally:
                    loop.close()
            
            # Run the OpenAI API call in an executor to prevent blocking I/O
            response: ChatCompletion = await self.hass.async_add_executor_job(execute_openai_request)
            # On success, cache the token param that was used
            self._token_param_cache[str(model)] = list(token_kwargs.keys())[0]
            
            # Log the API interaction to a temp file
            response_dict = response.model_dump(exclude_none=True)
            log_openai_interaction(self.hass, request_data, response_dict)
        except OpenAIError as err:
            _LOGGER.error("OpenAI API error: %s", err)
            raise

        _LOGGER.info("Response %s", json.dumps(response.model_dump(exclude_none=True)))

        if response.usage.total_tokens > context_threshold:
            await self.truncate_message_history(messages, exposed_entities, user_input)

        choice: Choice = response.choices[0]
        message = choice.message

        if choice.finish_reason == "function_call":
            _LOGGER.info("Function call detected: %s", message.function_call.name)
            return await self.execute_function_call(
                user_input, messages, message, exposed_entities, n_requests + 1
            )
        if choice.finish_reason == "tool_calls":
            tool_call_names = [tool.function.name for tool in message.tool_calls]
            _LOGGER.info("Tool calls detected: %s", ", ".join(tool_call_names))
            return await self.execute_tool_calls(
                user_input, messages, message, exposed_entities, n_requests + 1
            )
        if choice.finish_reason == "length":
            # If we used max_tokens, some GPT-5 style backends require max_completion_tokens instead.
            # Retry once swapping the token parameter (same numeric cap), then cache the result.
            used_token_param = list(token_kwargs.keys())[0]
            if used_token_param == "max_tokens":
                alt_param = "max_completion_tokens"
                alt_token_value = int(list(token_kwargs.values())[0])  # Ensure integer
                _LOGGER.debug(
                    "Finish reason 'length' with %s=%s; retrying once with %s",
                    used_token_param,
                    alt_token_value,
                    alt_param,
                )

                def execute_retry_swap_param() -> ChatCompletion:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(
                            self.client.chat.completions.create(
                                model=model,
                                messages=messages,
                                user=user_input.conversation_id,
                                **tool_kwargs,
                                **sampler_kwargs,
                                **{alt_param: alt_token_value},
                            )
                        )
                    finally:
                        loop.close()

                response_retry: ChatCompletion = await self.hass.async_add_executor_job(execute_retry_swap_param)
                # Cache discovered capability
                self._token_param_cache[str(model)] = alt_param
                response = response_retry
                choice = response.choices[0]
                message = choice.message
                if choice.finish_reason == "length":
                    # If swap fails at transport level, surface the original condition
                    raise TokenLengthExceededError(response.usage.completion_tokens)
            else:
                raise TokenLengthExceededError(response.usage.completion_tokens)

        return OpenAIQueryResponse(response=response, message=message)

    async def execute_function_call(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
    ) -> OpenAIQueryResponse:
        function_name = message.function_call.name
        function = next(
            (s for s in self.get_functions() if s["spec"]["name"] == function_name),
            None,
        )
        if function is not None:
            return await self.execute_function(
                user_input,
                messages,
                message,
                exposed_entities,
                n_requests,
                function,
            )
        raise FunctionNotFound(function_name)

    async def execute_function(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
        function,
    ) -> OpenAIQueryResponse:
        # First append the assistant message with function_call to the conversation history
        messages.append(message.model_dump(exclude_none=True))
        
        function_executor = get_function_executor(function["function"]["type"])

        try:
            arguments = json.loads(message.function_call.arguments)
        except json.decoder.JSONDecodeError as err:
            raise ParseArgumentsFailed(message.function_call.arguments) from err

        # Execute the function
        result = await function_executor.execute(
            self.hass, function["function"], arguments, user_input, exposed_entities
        )

        # Create a function response message
        function_message = {
            "role": "function",
            "name": message.function_call.name,
            "content": str(result),
        }
        
        # Append the function response to conversation history
        messages.append(function_message)
        
        _LOGGER.debug(
            "Added function response for %s: %s",
            message.function_call.name,
            str(result),
        )
        
        # Continue the conversation with the updated message history
        return await self.query(user_input, messages, exposed_entities, n_requests)

    async def execute_tool_calls(
        self,
        user_input: conversation.ConversationInput,
        messages,
        message: ChatCompletionMessage,
        exposed_entities,
        n_requests,
    ) -> OpenAIQueryResponse:
        # First append the assistant message with tool_calls to the conversation history
        messages.append(message.model_dump(exclude_none=True))
        
        # Process each tool call sequentially
        for tool in message.tool_calls:
            function_name = tool.function.name
            function = next(
                (s for s in self.get_functions() if s["spec"]["name"] == function_name),
                None,
            )
            
            if function is not None:
                try:
                    # Execute the tool function
                    result = await self.execute_tool_function(
                        user_input,
                        tool,
                        exposed_entities,
                        function,
                    )
                    
                    # Create a tool response message with the correct tool_call_id
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool.id,
                        "name": function_name,
                        "content": result,
                    }
                except (EntityNotExposed, EntityNotFound, HomeAssistantError) as err:
                    # Catch entity-related and other HA errors and provide a proper tool response
                    # instead of propagating the exception
                    _LOGGER.error("Error executing tool call %s: %s", function_name, err)
                    
                    # Create a tool response message with the error
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool.id,
                        "name": function_name,
                        "content": json.dumps({"error": str(err)}),
                    }
                
                # Append the tool response to conversation history
                messages.append(tool_message)
                
                _LOGGER.debug(
                    "Added tool response for %s (id: %s): %s",
                    function_name,
                    tool.id,
                    tool_message["content"],
                )
            else:
                # Create an error response for unknown functions instead of raising an exception
                error_message = f"function '{function_name}' does not exist"
                _LOGGER.error(error_message)
                
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool.id,
                    "name": function_name,
                    "content": json.dumps({"error": error_message}),
                }
                messages.append(tool_message)
                
        # Continue the conversation with the updated message history
        _LOGGER.debug("Messages after tool calls: %s", json.dumps(messages[-5:]))
        return await self.query(user_input, messages, exposed_entities, n_requests)

    async def execute_tool_function(
        self,
        user_input: conversation.ConversationInput,
        tool,
        exposed_entities,
        function,
    ) -> str:
        function_executor = get_function_executor(function["function"]["type"])

        try:
            arguments = json.loads(tool.function.arguments)
        except json.decoder.JSONDecodeError as err:
            raise ParseArgumentsFailed(tool.function.arguments) from err

        result = await function_executor.execute(
            self.hass, function["function"], arguments, user_input, exposed_entities
        )
        return str(result)


class OpenAIQueryResponse:
    """OpenAI query response value object."""

    def __init__(
        self, response: ChatCompletion, message: ChatCompletionMessage
    ) -> None:
        """Initialize OpenAI query response value object."""
        self.response = response
        self.message = message

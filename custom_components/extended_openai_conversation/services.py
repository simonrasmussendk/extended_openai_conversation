import base64
import logging
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

from openai import AsyncOpenAI
from openai import OpenAIError
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
)
import voluptuous as vol

from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.typing import ConfigType

from .const import (
    DOMAIN,
    SERVICE_QUERY_IMAGE,
    SERVICE_CLEAR_CONVERSATIONS,
    DATA_CONVERSATION_STORE,
)
from .helpers import (
    create_openai_client,
    log_openai_interaction,
    resolve_token_parameters,
    build_sampler_kwargs,
    async_get_preset_for_model,
    extract_and_validate_model_params,
    get_preferred_token_param,
)

def create_service_schemas():
    """Create service schemas dynamically."""
    config_entry_selector = selector.ConfigEntrySelector({"integration": DOMAIN})
    
    return {
        "query_image": vol.Schema({
            vol.Required("config_entry"): config_entry_selector,
            vol.Required("model", default="gpt-4-vision-preview"): cv.string,
            vol.Required("prompt"): cv.string,
            vol.Required("images"): vol.All(cv.ensure_list, [{"url": cv.string}]),
            vol.Optional("max_tokens", default=300): cv.positive_int,
        }),
        "clear_conversations": vol.Schema({
            vol.Optional("conversation_id"): cv.string,
        }),
    }


# Create schemas at module level
_SERVICE_SCHEMAS = create_service_schemas()
QUERY_IMAGE_SCHEMA = _SERVICE_SCHEMAS["query_image"]
CLEAR_CONVERSATIONS_SCHEMA = _SERVICE_SCHEMAS["clear_conversations"]

_LOGGER = logging.getLogger(__package__)


async def async_setup_services(hass: HomeAssistant, config: ConfigType) -> None:
    """Set up services for the extended openai conversation component."""

    async def query_image(call: ServiceCall) -> ServiceResponse:
        """Query an image."""
        try:
            entry = hass.config_entries.async_get_entry(call.data["config_entry"])
            params = extract_and_validate_model_params(entry.options)
            model = params["model"]
            max_tokens = params["max_tokens"]
            temperature = params["temperature"]
            top_p = params["top_p"]

            images = [
                {"type": "image_url", "image_url": to_image_param(hass, image)}
                for image in call.data["images"]
            ]

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": call.data["prompt"]}] + images,
                }
            ]
            _LOGGER.info("Prompt for %s: %s", model, messages)

            # Resolve preset and token parameter name with optional clamping (non-blocking)
            preset = await async_get_preset_for_model(hass, model)
            
            # Use centralized token parameter resolution
            agent = hass.data[DOMAIN].get(entry.entry_id)
            token_cache = getattr(agent, "_token_param_cache", {}) if agent else {}
            token_kwargs, _ = resolve_token_parameters(
                model=model,
                max_tokens=max_tokens,
                preset=preset,
                token_param_cache=token_cache,
            )
            
            # Build sampler parameters
            sampler_kwargs = build_sampler_kwargs(model, temperature, top_p, preset)
            
            # Prepare API request parameters
            request_params = {
                "model": model,
                "messages": messages,
                **token_kwargs,
                **sampler_kwargs,
            }

            client = create_openai_client(hass, entry)
            try:
                response = await client.chat.completions.create(**request_params)
            except Exception as err:
                # Fallback if server rejects the chosen token parameter
                err_str = str(err)
                def _is_unsupported(param: str) -> bool:
                    return "Unsupported parameter" in err_str and f"'{param}'" in err_str
                alt_param = "max_tokens" if "max_completion_tokens" in request_params else "max_completion_tokens"
                if _is_unsupported("max_completion_tokens") or _is_unsupported("max_tokens"):
                    alt_token_kwargs = {alt_param: max_tokens}
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **alt_token_kwargs,
                    )
                else:
                    raise
            response_dict = response.model_dump()
            _LOGGER.info("Response %s", response_dict)
        except OpenAIError as err:
            raise HomeAssistantError(f"Error generating image: {err}") from err

        return response_dict

    async def clear_conversations(call: ServiceCall) -> None:
        """Clear conversations from the memory store."""
        conversation_store = hass.data[DOMAIN].get(DATA_CONVERSATION_STORE)
        if not conversation_store:
            _LOGGER.warning("Conversation store not found")
            return

        conversation_id = call.data.get("conversation_id")
        if conversation_id:
            # Clear a specific conversation
            existing_conversation = conversation_store.get_conversation(conversation_id)
            if existing_conversation:
                # Save an empty conversation to effectively remove it
                conversation_store.save_conversation(conversation_id, [])
                _LOGGER.info(f"Cleared conversation with ID: {conversation_id}")
            else:
                _LOGGER.warning(f"Conversation with ID {conversation_id} not found")
        else:
            # Clear all conversations
            conversation_store.clear_all()
            _LOGGER.info("All conversations cleared")

    hass.services.async_register(
        DOMAIN,
        SERVICE_QUERY_IMAGE,
        query_image,
        schema=QUERY_IMAGE_SCHEMA,
        supports_response=SupportsResponse.ONLY,
    )

    # Register the clear conversations service
    hass.services.async_register(
        DOMAIN,
        SERVICE_CLEAR_CONVERSATIONS,
        clear_conversations,
        schema=CLEAR_CONVERSATIONS_SCHEMA,
    )


def to_image_param(hass: HomeAssistant, image) -> ChatCompletionContentPartImageParam:
    """Convert url to base64 encoded image if local."""
    url = image["url"]

    if urlparse(url).scheme in cv.EXTERNAL_URL_PROTOCOL_SCHEMA_LIST:
        return image

    if not hass.config.is_allowed_path(url):
        raise HomeAssistantError(
            f"Cannot read `{url}`, no access to path; "
            "`allowlist_external_dirs` may need to be adjusted in "
            "`configuration.yaml`"
        )
    if not Path(url).exists():
        raise HomeAssistantError(f"`{url}` does not exist")
    mime_type, _ = mimetypes.guess_type(url)
    if mime_type is None or not mime_type.startswith("image"):
        raise HomeAssistantError(f"`{url}` is not an image")

    image["url"] = f"data:{mime_type};base64,{encode_image(url)}"
    return image


def encode_image(image_path):
    """Convert to base64 encoded image."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

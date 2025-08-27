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
    async_get_preset_for_model,
    get_param_names,
    get_limits,
)

QUERY_IMAGE_SCHEMA = vol.Schema(
    {
        vol.Required("config_entry"): selector.ConfigEntrySelector(
            {
                "integration": DOMAIN,
            }
        ),
        vol.Required("model", default="gpt-4-vision-preview"): cv.string,
        vol.Required("prompt"): cv.string,
        vol.Required("images"): vol.All(cv.ensure_list, [{"url": cv.string}]),
        vol.Optional("max_tokens", default=300): cv.positive_int,
    }
)

CLEAR_CONVERSATIONS_SCHEMA = vol.Schema(
    {
        vol.Optional("conversation_id"): cv.string,
    }
)

_LOGGER = logging.getLogger(__package__)


async def async_setup_services(hass: HomeAssistant, config: ConfigType) -> None:
    """Set up services for the extended openai conversation component."""

    async def query_image(call: ServiceCall) -> ServiceResponse:
        """Query an image."""
        try:
            model = call.data["model"]
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
            preset_param_names = get_param_names(preset) if preset else set()
            limits = get_limits(preset) if preset else {}

            # Choose token parameter name: preset -> heuristic
            if "max_completion_tokens" in preset_param_names:
                token_param_name = "max_completion_tokens"
            elif "max_tokens" in preset_param_names:
                token_param_name = "max_tokens"
            else:
                token_param_name = "max_completion_tokens" if "gpt-5" in str(model).lower() else "max_tokens"

            # Clamp based on preset limits when defined
            max_tokens_val = call.data["max_tokens"]
            limit_val = limits.get(token_param_name)
            if isinstance(limit_val, int) and isinstance(max_tokens_val, int) and max_tokens_val > limit_val:
                _LOGGER.debug("Clamping %s from %s to preset limit %s", token_param_name, max_tokens_val, limit_val)
                max_tokens_val = limit_val

            token_kwargs = {token_param_name: max_tokens_val}

            client = AsyncOpenAI(
                api_key=hass.data[DOMAIN][call.data["config_entry"]]["api_key"]
            )
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **token_kwargs,
                )
            except Exception as err:
                # Fallback if server rejects the chosen token parameter
                err_str = str(err)
                def _is_unsupported(param: str) -> bool:
                    return "Unsupported parameter" in err_str and f"'{param}'" in err_str
                alt_param = "max_tokens" if token_param_name == "max_completion_tokens" else "max_completion_tokens"
                if _is_unsupported(token_param_name):
                    alt_token_kwargs = {alt_param: call.data["max_tokens"]}
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

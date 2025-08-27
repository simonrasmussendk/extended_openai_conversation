"""Config flow for OpenAI Conversation integration."""
from __future__ import annotations

import logging
import types
from types import MappingProxyType
from typing import Any

from openai import APIConnectionError, AuthenticationError
import voluptuous as vol
import yaml

from homeassistant import config_entries
from homeassistant.const import CONF_API_KEY, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)

from .const import (
    CONF_API_VERSION,
    CONF_ATTACH_USERNAME,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_CONVERSATION_EXPIRATION_TIME,
    CONF_DOMAIN_KEYWORDS,
    CONF_ENABLE_STT,
    CONF_FUNCTIONS,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_ORGANIZATION,
    CONF_PROMPT,
    CONF_SKIP_AUTHENTICATION,
    CONF_STT_API_KEY,
    CONF_STT_API_VERSION,
    CONF_STT_BASE_URL,
    CONF_STT_LANGUAGE,
    CONF_STT_MODEL,
    CONF_STT_ORGANIZATION,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_USE_TOOLS,
    CONTEXT_TRUNCATE_STRATEGIES,
    DEFAULT_ATTACH_USERNAME,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONF_BASE_URL,
    DEFAULT_CONF_FUNCTIONS,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_CONVERSATION_EXPIRATION_TIME,
    DEFAULT_DOMAIN_KEYWORDS,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_NAME,
    DEFAULT_PROMPT,
    DEFAULT_SKIP_AUTHENTICATION,
    DEFAULT_STT_LANGUAGE,
    DEFAULT_STT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_USE_TOOLS,
    DOMAIN,
)
from .helpers import validate_authentication, load_presets

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_NAME): str,
        vol.Required(CONF_API_KEY): str,
        vol.Optional(CONF_BASE_URL, default=DEFAULT_CONF_BASE_URL): str,
        vol.Optional(CONF_API_VERSION): str,
        vol.Optional(CONF_ORGANIZATION): str,
        vol.Optional(
            CONF_SKIP_AUTHENTICATION, default=DEFAULT_SKIP_AUTHENTICATION
        ): bool,
    }
)

DEFAULT_CONF_FUNCTIONS_STR = yaml.dump(DEFAULT_CONF_FUNCTIONS, sort_keys=False)

DEFAULT_OPTIONS = types.MappingProxyType(
    {
        CONF_PROMPT: DEFAULT_PROMPT,
        CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
        CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
        CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION: DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
        CONF_TOP_P: DEFAULT_TOP_P,
        CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
        CONF_FUNCTIONS: DEFAULT_CONF_FUNCTIONS_STR,
        CONF_ATTACH_USERNAME: DEFAULT_ATTACH_USERNAME,
        CONF_USE_TOOLS: DEFAULT_USE_TOOLS,
        CONF_CONTEXT_THRESHOLD: DEFAULT_CONTEXT_THRESHOLD,
        CONF_CONTEXT_TRUNCATE_STRATEGY: DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
        CONF_DOMAIN_KEYWORDS: DEFAULT_DOMAIN_KEYWORDS,
        CONF_ENABLE_STT: False,
        CONF_STT_API_KEY: "",
        CONF_STT_BASE_URL: "",
        CONF_STT_API_VERSION: "",
        CONF_STT_ORGANIZATION: "",
        CONF_STT_MODEL: DEFAULT_STT_MODEL,
        CONF_STT_LANGUAGE: DEFAULT_STT_LANGUAGE,
    }
)


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    api_key = data[CONF_API_KEY]
    base_url = data.get(CONF_BASE_URL)
    api_version = data.get(CONF_API_VERSION)
    organization = data.get(CONF_ORGANIZATION)
    skip_authentication = data.get(CONF_SKIP_AUTHENTICATION)

    if base_url == DEFAULT_CONF_BASE_URL:
        # Do not set base_url if using OpenAI for case of OpenAI's base_url change
        base_url = None
        data.pop(CONF_BASE_URL)

    await validate_authentication(
        hass=hass,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
        organization=organization,
        skip_authentication=skip_authentication,
    )


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors = {}

        try:
            await validate_input(self.hass, user_input)
        except APIConnectionError:
            errors["base"] = "cannot_connect"
        except AuthenticationError:
            errors["base"] = "invalid_auth"
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            return self.async_create_entry(
                title=user_input.get(CONF_NAME, DEFAULT_NAME), data=user_input
            )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Create the options flow."""
        return OptionsFlow(config_entry)


class OptionsFlow(config_entries.OptionsFlow):
    """OpenAI config flow options handler."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self._config_entry = config_entry
        self._selected_preset: str | None = None
        self._presets: dict[str, dict[str, Any]] = {}

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Entry point -> redirect to preset selection."""
        return await self.async_step_preset()

    async def async_step_preset(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Step 1: Select a model preset."""
        # Load presets each time to reflect user file changes (offload I/O)
        presets_list = await self.hass.async_add_executor_job(load_presets, self.hass)
        # Convert list[dict] -> dict[key]->preset for easier lookups
        self._presets = {
            p.get("key"): p
            for p in (presets_list or [])
            if isinstance(p, dict) and p.get("key")
        }

        # If no presets available, fall back to legacy single-step options
        if not self._presets:
            return await self.async_step_options()

        existing = self._config_entry.options or {}
        current_model = existing.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        default_model = (
            current_model
            if current_model in self._presets
            else next(iter(self._presets.keys()))
        )

        selector = SelectSelector(
            SelectSelectorConfig(
                options=[
                    SelectOptionDict(value=key, label=preset.get("label", key))
                    for key, preset in self._presets.items()
                ],
                mode=SelectSelectorMode.DROPDOWN,
            )
        )

        # Use field key "model" to align with existing translations
        data_schema = vol.Schema({vol.Required("model", default=default_model): selector})

        if user_input is not None:
            # Map translated field key back to CONF_CHAT_MODEL
            self._selected_preset = user_input.get("model", default_model)
            return await self.async_step_options()

        return self.async_show_form(step_id="preset", data_schema=data_schema)

    async def async_step_options(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Step 2: Dynamic options for selected preset + general settings."""
        if not self._presets:
            presets_list = await self.hass.async_add_executor_job(load_presets, self.hass)
            self._presets = {
                p.get("key"): p
                for p in (presets_list or [])
                if isinstance(p, dict) and p.get("key")
            }

        # Resolve preset
        preset_key = self._selected_preset or self._config_entry.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        preset = self._presets.get(preset_key)

        if user_input is not None:
            # Merge with existing options to preserve unseen keys
            merged = dict(DEFAULT_OPTIONS)
            merged.update(dict(self._config_entry.options or {}))
            merged.update(user_input)
            merged[CONF_CHAT_MODEL] = preset_key
            return self.async_create_entry(title=merged.get(CONF_NAME, DEFAULT_NAME), data=merged)

        # Build dynamic schema
        data_schema_dict: dict[Any, Any] = {}
        options = self._config_entry.options or {}

        # Add preset-specific parameters
        if isinstance(preset, dict):
            for p in preset.get("parameters", []) or []:
                if not isinstance(p, dict):
                    continue
                name = p.get("name")
                if not name:
                    continue
                p_type = p.get("type", "number")
                p_default = options.get(name, p.get("default"))
                if p_type == "select":
                    opts = p.get("options", []) or []
                    data_schema_dict[
                        vol.Optional(
                            name,
                            description={"suggested_value": p_default},
                            default=p.get("default"),
                        )
                    ] = SelectSelector(
                        SelectSelectorConfig(
                            options=[SelectOptionDict(value=str(v), label=str(v)) for v in opts],
                            mode=SelectSelectorMode.DROPDOWN,
                        )
                    )
                else:
                    # number (and generic numeric)
                    min_v = p.get("min", 0)
                    max_v = p.get("max", 1)
                    step_v = p.get("step", 1)
                    data_schema_dict[
                        vol.Optional(
                            name,
                            description={"suggested_value": p_default},
                            default=p.get("default"),
                        )
                    ] = NumberSelector(NumberSelectorConfig(min=min_v, max=max_v, step=step_v))

        # General options not tied to preset
        data_schema_dict.update(
            {
                vol.Optional(
                    CONF_PROMPT,
                    description={"suggested_value": options.get(CONF_PROMPT, DEFAULT_PROMPT)},
                    default=DEFAULT_PROMPT,
                ): TemplateSelector(),
                vol.Optional(
                    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                    description={
                        "suggested_value": options.get(
                            CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                            DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                        )
                    },
                    default=DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                ): int,
                vol.Optional(
                    CONF_FUNCTIONS,
                    description={"suggested_value": options.get(CONF_FUNCTIONS, DEFAULT_CONF_FUNCTIONS_STR)},
                    default=DEFAULT_CONF_FUNCTIONS_STR,
                ): TemplateSelector(),
                vol.Optional(
                    CONF_ATTACH_USERNAME,
                    description={"suggested_value": options.get(CONF_ATTACH_USERNAME, DEFAULT_ATTACH_USERNAME)},
                    default=DEFAULT_ATTACH_USERNAME,
                ): BooleanSelector(),
                vol.Optional(
                    CONF_USE_TOOLS,
                    description={"suggested_value": options.get(CONF_USE_TOOLS, DEFAULT_USE_TOOLS)},
                    default=DEFAULT_USE_TOOLS,
                ): BooleanSelector(),
                vol.Optional(
                    CONF_CONTEXT_THRESHOLD,
                    description={"suggested_value": options.get(CONF_CONTEXT_THRESHOLD, DEFAULT_CONTEXT_THRESHOLD)},
                    default=DEFAULT_CONTEXT_THRESHOLD,
                ): int,
                vol.Optional(
                    CONF_CONTEXT_TRUNCATE_STRATEGY,
                    description={"suggested_value": options.get(CONF_CONTEXT_TRUNCATE_STRATEGY, DEFAULT_CONTEXT_TRUNCATE_STRATEGY)},
                    default=DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=[
                            SelectOptionDict(value=strategy["key"], label=strategy["label"]) for strategy in CONTEXT_TRUNCATE_STRATEGIES
                        ],
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    CONF_DOMAIN_KEYWORDS,
                    description={"suggested_value": options.get(CONF_DOMAIN_KEYWORDS, DEFAULT_DOMAIN_KEYWORDS)},
                    default=DEFAULT_DOMAIN_KEYWORDS,
                ): TemplateSelector(),
                # STT options
                vol.Optional(
                    CONF_ENABLE_STT,
                    description={"suggested_value": options.get(CONF_ENABLE_STT, False)},
                    default=options.get(CONF_ENABLE_STT, False),
                ): BooleanSelector(),
                vol.Optional(
                    CONF_STT_API_KEY,
                    description={"suggested_value": options.get(CONF_STT_API_KEY, "")},
                    default=options.get(CONF_STT_API_KEY, ""),
                ): str,
                vol.Optional(
                    CONF_STT_BASE_URL,
                    description={"suggested_value": options.get(CONF_STT_BASE_URL, "")},
                    default=options.get(CONF_STT_BASE_URL, ""),
                ): str,
                vol.Optional(
                    CONF_STT_API_VERSION,
                    description={"suggested_value": options.get(CONF_STT_API_VERSION, "")},
                    default=options.get(CONF_STT_API_VERSION, ""),
                ): str,
                vol.Optional(
                    CONF_STT_ORGANIZATION,
                    description={"suggested_value": options.get(CONF_STT_ORGANIZATION, "")},
                    default=options.get(CONF_STT_ORGANIZATION, ""),
                ): str,
                vol.Required(
                    CONF_STT_MODEL,
                    description={"suggested_value": options.get(CONF_STT_MODEL, DEFAULT_STT_MODEL)},
                    default=options.get(CONF_STT_MODEL, DEFAULT_STT_MODEL),
                ): str,
                vol.Required(
                    CONF_STT_LANGUAGE,
                    description={"suggested_value": options.get(CONF_STT_LANGUAGE, DEFAULT_STT_LANGUAGE)},
                    default=options.get(CONF_STT_LANGUAGE, DEFAULT_STT_LANGUAGE),
                ): str,
            }
        )

        # Show dynamic options step
        return self.async_show_form(step_id="options", data_schema=vol.Schema(data_schema_dict))

    def openai_config_option_schema(self, options: dict[str, Any]) -> dict:
        """Return a schema for OpenAI completion options."""
        if not options:
            options = DEFAULT_OPTIONS

        return {
            vol.Optional(
                CONF_PROMPT,
                description={"suggested_value": options[CONF_PROMPT]},
                default=DEFAULT_PROMPT,
            ): TemplateSelector(),
            vol.Optional(
                CONF_CHAT_MODEL,
                description={
                    # New key in HA 2023.4
                    "suggested_value": options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
                },
                default=DEFAULT_CHAT_MODEL,
            ): str,
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": options[CONF_MAX_TOKENS]},
                default=DEFAULT_MAX_TOKENS,
            ): int,
            vol.Optional(
                CONF_TOP_P,
                description={"suggested_value": options[CONF_TOP_P]},
                default=DEFAULT_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": options[CONF_TEMPERATURE]},
                default=DEFAULT_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
                description={
                    "suggested_value": options[CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION]
                },
                default=DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
            ): int,
            vol.Optional(
                CONF_FUNCTIONS,
                description={"suggested_value": options.get(CONF_FUNCTIONS)},
                default=DEFAULT_CONF_FUNCTIONS_STR,
            ): TemplateSelector(),
            vol.Optional(
                CONF_ATTACH_USERNAME,
                description={"suggested_value": options.get(CONF_ATTACH_USERNAME)},
                default=DEFAULT_ATTACH_USERNAME,
            ): BooleanSelector(),
            vol.Optional(
                CONF_USE_TOOLS,
                description={"suggested_value": options.get(CONF_USE_TOOLS)},
                default=DEFAULT_USE_TOOLS,
            ): BooleanSelector(),
            vol.Optional(
                CONF_CONTEXT_THRESHOLD,
                description={"suggested_value": options.get(CONF_CONTEXT_THRESHOLD)},
                default=DEFAULT_CONTEXT_THRESHOLD,
            ): int,
            vol.Optional(
                CONF_CONTEXT_TRUNCATE_STRATEGY,
                description={
                    "suggested_value": options.get(CONF_CONTEXT_TRUNCATE_STRATEGY)
                },
                default=DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value=strategy["key"], label=strategy["label"])
                        for strategy in CONTEXT_TRUNCATE_STRATEGIES
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_DOMAIN_KEYWORDS,
                description={"suggested_value": options.get(CONF_DOMAIN_KEYWORDS)},
                default=DEFAULT_DOMAIN_KEYWORDS,
            ): TemplateSelector(),
            # STT options
            vol.Optional(
                CONF_ENABLE_STT,
                description={"suggested_value": options.get(CONF_ENABLE_STT, False)},
                default=False,
            ): BooleanSelector(),
            vol.Optional(
                CONF_STT_API_KEY,
                description={"suggested_value": options.get(CONF_STT_API_KEY, "")},
                default="",
            ): str,
            vol.Optional(
                CONF_STT_BASE_URL,
                description={"suggested_value": options.get(CONF_STT_BASE_URL, "")},
                default="",
            ): str,
            vol.Optional(
                CONF_STT_API_VERSION,
                description={"suggested_value": options.get(CONF_STT_API_VERSION, "")},
                default="",
            ): str,
            vol.Optional(
                CONF_STT_ORGANIZATION,
                description={"suggested_value": options.get(CONF_STT_ORGANIZATION, "")},
                default="",
            ): str,
            vol.Required(
                CONF_STT_MODEL,
                description={"suggested_value": options.get(CONF_STT_MODEL, DEFAULT_STT_MODEL)},
                default=DEFAULT_STT_MODEL,
            ): str,
            vol.Required(
                CONF_STT_LANGUAGE,
                description={"suggested_value": options.get(CONF_STT_LANGUAGE, DEFAULT_STT_LANGUAGE)},
                default=DEFAULT_STT_LANGUAGE,
            ): str,
        }

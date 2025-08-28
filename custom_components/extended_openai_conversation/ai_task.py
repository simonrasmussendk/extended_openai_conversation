"""AI Task platform for Extended OpenAI Conversation integration."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

from openai import OpenAIError

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.components.ai_task import (
    AITaskEntity,
    AITaskEntityFeature,
    GenDataTask,
    GenDataTaskResult,
)

from .const import (
    DOMAIN,
    DEFAULT_NAME,
    CONF_CHAT_MODEL,
    DEFAULT_CHAT_MODEL,
    CONF_MAX_TOKENS,
    DEFAULT_MAX_TOKENS,
    CONF_TEMPERATURE,
    DEFAULT_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_TOP_P,
)
from .helpers import (
    async_get_preset_for_model,
    get_param_names,
    get_limits,
    log_openai_interaction,
    resolve_token_parameters,
    build_sampler_kwargs,
)
from . import DATA_AGENT  # hass.data key set in __init__

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> bool:
    """Set up the AI Task entity."""
    async_add_entities([ExtendedOpenAIAITaskEntity(hass, entry)])
    _LOGGER.info("Extended OpenAI AI Task entity added for entry %s", entry.entry_id)
    return True


class ExtendedOpenAIAITaskEntity(AITaskEntity):
    """AI Task entity powered by the Extended OpenAI Conversation backend."""

    _attr_supported_features = AITaskEntityFeature.GENERATE_DATA

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__()
        self.hass = hass
        self.entry = entry
        self._attr_unique_id = f"{entry.entry_id}_ai_task"
        entry_title = getattr(entry, "title", None) or DEFAULT_NAME
        self._attr_name = f"{entry_title} AI Task"

    async def _ensure_agent_ready(self):
        """Ensure the shared OpenAI agent and client are initialized."""
        data = self.hass.data.get(DOMAIN, {}).get(self.entry.entry_id, {})
        agent = data.get(DATA_AGENT)
        if agent is None:
            raise RuntimeError("OpenAI agent not available; integration not initialized")
        if getattr(agent, "client", None) is None:
            # Wait for lazy initialization to complete
            try:
                await agent._initialize_client_task
            except Exception as err:  # noqa: BLE001
                raise RuntimeError(f"Failed to initialize OpenAI client: {err}") from err
        return agent

    @property
    def supported_features(self) -> AITaskEntityFeature:
        return self._attr_supported_features

    async def _async_generate_data(
        self, task: GenDataTask, chat_log: Any
    ) -> GenDataTaskResult:
        """Handle a generate data task using the OpenAI chat completions API.

        If task.structure is provided, we will instruct the model to return a JSON object
        that matches the structure as closely as possible and try to parse it.
        """
        # Ensure agent is ready; surface errors as structured result instead of raising
        try:
            agent = await self._ensure_agent_ready()
        except Exception as err:  # noqa: BLE001
            _LOGGER.error("AI Task agent not ready: %s", err, exc_info=err)
            return GenDataTaskResult(conversation_id=None, data={"error": str(err)})

        # Build a system message using existing prompt + exposed entities
        try:
            exposed_entities = agent.get_exposed_entities()
            dummy_input = SimpleNamespace(device_id=None)
            system_message = agent._generate_system_message(exposed_entities, dummy_input)
        except Exception as err:  # noqa: BLE001
            _LOGGER.debug("Falling back to minimal system prompt: %s", err)
            system_message = {
                "role": "system",
                "content": (
                    "You are an assistant for Home Assistant. Generate concise, helpful results."
                ),
            }

        instructions: str = getattr(task, "instructions", "") or ""
        structure: Any = getattr(task, "structure", None)
        conversation_id: str | None = getattr(chat_log, "conversation_id", None)

        if structure:
            # Encourage strict JSON output aligned to the provided structure
            try:
                structure_str = json.dumps(structure, ensure_ascii=False, default=str)
            except Exception as dump_err:  # noqa: BLE001
                _LOGGER.warning(
                    "Failed to serialize structure for prompt: %s; falling back to keys only",
                    dump_err,
                )
                try:
                    keys_only = {"fields": list(getattr(structure, "keys", lambda: [])())}
                except Exception:  # noqa: BLE001
                    keys_only = {"fields": []}
                structure_str = json.dumps(keys_only, ensure_ascii=False)

            user_prompt = (
                "Generate data according to these instructions. Return a single JSON object only, "
                "with keys matching the provided structure. Do not include any extra text or code fencing.\n\n"
                f"Instructions:\n{instructions}\n\nStructure (example selectors for UI; use as schema guidance):\n{structure_str}\n"
            )
            response_format = {"type": "json_object"}
        else:
            user_prompt = instructions
            response_format = None

        messages = [system_message, {"role": "user", "content": user_prompt}]

        # Read parameters and prepare request; surface errors instead of raising
        try:
            # Read parameters from options
            opts = {**self.entry.options}
            model = opts.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
            # Ensure int for max tokens
            max_tokens_opt = opts.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
            try:
                max_tokens = int(max_tokens_opt)
            except (TypeError, ValueError):
                try:
                    max_tokens = int(float(max_tokens_opt))
                except (TypeError, ValueError):
                    max_tokens = int(DEFAULT_MAX_TOKENS)
            temperature = opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
            top_p = opts.get(CONF_TOP_P, DEFAULT_TOP_P)

            # Derive parameter behavior from preset and model heuristics
            is_gpt5 = "gpt-5" in str(model).lower()
            preset = await async_get_preset_for_model(self.hass, model)
            preset_param_names = get_param_names(preset) if preset else set()

            # Use centralized token parameter resolution
            agent_cache = getattr(agent, "_token_param_cache", {})
            token_kwargs, token_param_name = resolve_token_parameters(
                model=model,
                max_tokens=max_tokens,
                preset=preset,
                token_param_cache=agent_cache,
            )

            # Build sampler kwargs using centralized logic
            sampler_kwargs = build_sampler_kwargs(
                model=model,
                temperature=temperature,
                top_p=top_p,
                preset=preset,
            )

            # Optional reasoning mapping if preset declares reasoning_effort
            reasoning_kwargs: dict[str, Any] = {}
            if "reasoning_effort" in preset_param_names:
                effort = opts.get("reasoning_effort")
                if effort:
                    reasoning_kwargs = {"reasoning": {"effort": effort}}

            # Prepare request payload
            tool_kwargs: dict[str, Any] = {}
            base_kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "user": conversation_id,
                **tool_kwargs,
            }
            # Response format for structured output
            if response_format is not None:
                base_kwargs["response_format"] = response_format

            # For logging
            request_data = {
                "model": model,
                "messages": messages,
                token_param_name: max_tokens,
                "user": conversation_id,
                **sampler_kwargs,
                **reasoning_kwargs,
                **tool_kwargs,
            }
            if "response_format" in base_kwargs:
                request_data["response_format"] = base_kwargs["response_format"]
        except Exception as err:  # noqa: BLE001
            _LOGGER.error("AI Task parameter resolution failed: %s", err, exc_info=err)
            return GenDataTaskResult(conversation_id=conversation_id, data={"error": str(err)})

        # Execute request with retries mirroring conversation agent behavior
        try:
            try:
                resp = await agent.client.chat.completions.create(
                    **base_kwargs,
                    **token_kwargs,
                    **sampler_kwargs,
                    **reasoning_kwargs,
                )
            except Exception as err:  # noqa: BLE001
                err_str = str(err)
                # Drop reasoning if unsupported by SDK or server
                if isinstance(err, TypeError) and (
                    "unexpected keyword argument 'reasoning'" in err_str
                    or "got an unexpected keyword argument 'reasoning'" in err_str
                ):
                    resp = await agent.client.chat.completions.create(
                        **base_kwargs,
                        **token_kwargs,
                        **sampler_kwargs,
                    )
                else:
                    def _is_unsupported(param: str) -> bool:
                        return "Unsupported parameter" in err_str and f"'{param}'" in err_str
                    def _is_unsupported_value(param: str) -> bool:
                        return "Unsupported value" in err_str and f"'{param}'" in err_str
                    # If response_format unsupported, drop it and retry
                    if _is_unsupported("response_format") or _is_unsupported_value("response_format") or (
                        isinstance(err, TypeError)
                        and (
                            "unexpected keyword argument 'response_format'" in err_str
                            or "got an unexpected keyword argument 'response_format'" in err_str
                        )
                    ):
                        base_no_rf = dict(base_kwargs)
                        base_no_rf.pop("response_format", None)
                        request_data.pop("response_format", None)
                        resp = await agent.client.chat.completions.create(
                            **base_no_rf,
                            **token_kwargs,
                            **sampler_kwargs,
                            **reasoning_kwargs,
                        )
                    # Swap token parameter if unsupported
                    alt_param = "max_tokens" if token_param_name == "max_completion_tokens" else "max_completion_tokens"
                    if _is_unsupported(token_param_name):
                        alt_token_kwargs = {alt_param: max_tokens}
                        resp = await agent.client.chat.completions.create(
                            **base_kwargs,
                            **alt_token_kwargs,
                            **sampler_kwargs,
                            **reasoning_kwargs,
                        )
                        # Cache discovered capability
                        try:
                            agent._token_param_cache[str(model)] = alt_param  # type: ignore[attr-defined]
                        except Exception:  # noqa: BLE001
                            pass
                        # Update logging payload
                        request_data.pop(token_param_name, None)
                        request_data[alt_param] = max_tokens
                    # If reasoning unsupported at server level, drop it and retry
                    elif _is_unsupported("reasoning") or _is_unsupported_value("reasoning"):
                        resp = await agent.client.chat.completions.create(
                            **base_kwargs,
                            **token_kwargs,
                            **sampler_kwargs,
                        )
                    # If temperature unsupported, drop it; if then top_p also fails, drop both
                    elif _is_unsupported("temperature") or _is_unsupported_value("temperature"):
                        try:
                            resp = await agent.client.chat.completions.create(
                                **base_kwargs,
                                **token_kwargs,
                                **({k: v for k, v in sampler_kwargs.items() if k != "temperature"}),
                            )
                        except Exception as err2:  # noqa: BLE001
                            err2_str = str(err2)
                            if _is_unsupported("top_p") or ("'top_p'" in err2_str and ("Unsupported parameter" in err2_str or "Unsupported value" in err2_str)):
                                resp = await agent.client.chat.completions.create(
                                    **base_kwargs,
                                    **token_kwargs,
                                )
                            else:
                                raise
                    # If only top_p is unsupported, drop it
                    elif _is_unsupported("top_p") or _is_unsupported_value("top_p"):
                        resp = await agent.client.chat.completions.create(
                            **base_kwargs,
                            **token_kwargs,
                            **({k: v for k, v in sampler_kwargs.items() if k != "top_p"}),
                        )
                    else:
                        raise

            # Cache the token parameter used on success if not already cached
            try:
                agent._token_param_cache[str(model)] = list(token_kwargs.keys())[0]  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

            # Log request/response
            try:
                response_dict = resp.model_dump(exclude_none=True)
                log_openai_interaction(self.hass, request_data, response_dict)
            except Exception:  # noqa: BLE001
                pass

            # If response was cut due to length and we used max_tokens, retry once swapping param
            try:
                choice = resp.choices[0]
                if getattr(choice, "finish_reason", None) == "length":
                    used_param = list(token_kwargs.keys())[0]
                    if used_param == "max_tokens":
                        try:
                            resp_retry = await agent.client.chat.completions.create(
                                **base_kwargs,
                                **sampler_kwargs,
                                **{"max_completion_tokens": max_tokens},
                            )
                            try:
                                agent._token_param_cache[str(model)] = "max_completion_tokens"  # type: ignore[attr-defined]
                            except Exception:  # noqa: BLE001
                                pass
                            resp = resp_retry
                            choice = resp.choices[0]
                        except OpenAIError:
                            # If swap fails at transport level, keep original resp
                            pass
            except Exception:  # noqa: BLE001
                pass

            # Extract content
            content: str = ""
            try:
                # SDK types: choice.message.content is optional
                content = getattr(choice.message, "content", None) or ""
            except Exception:  # noqa: BLE001
                content = ""

            if not structure:
                return GenDataTaskResult(conversation_id=conversation_id, data=content)

            # Try to parse JSON if structure requested
            data_out: Any
            try:
                data_out = json.loads(content)
            except Exception:  # noqa: BLE001
                # Best-effort extraction of the first JSON object
                data_out = self._best_effort_extract_json(content)
                if data_out is None:
                    data_out = {"text": content}

            return GenDataTaskResult(conversation_id=conversation_id, data=data_out)
        except OpenAIError as err:
            _LOGGER.error("AI Task generation failed (OpenAI): %s", err)
            return GenDataTaskResult(conversation_id=conversation_id, data={"error": str(err)})
        except Exception as err:  # noqa: BLE001
            _LOGGER.error("AI Task generation failed: %s", err, exc_info=err)
            return GenDataTaskResult(conversation_id=conversation_id, data={"error": str(err)})

    @staticmethod
    def _best_effort_extract_json(text: str) -> Any | None:
        """Extract first top-level JSON object from text, if present."""
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:  # noqa: BLE001
                return None
        return None

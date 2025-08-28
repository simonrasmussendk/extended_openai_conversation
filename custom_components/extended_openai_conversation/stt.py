"""STT platform for Extended OpenAI Conversation integration."""
from __future__ import annotations

import asyncio
import logging
from io import BytesIO
import wave
from typing import Any, AsyncIterable

from openai import AsyncAzureOpenAI, AsyncOpenAI, OpenAIError

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    Provider,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.httpx_client import get_async_client

from .helpers import create_openai_client, extract_and_validate_model_params
from homeassistant.const import CONF_API_KEY
from .const import (
    CONF_API_VERSION,
    CONF_BASE_URL,
    CONF_ORGANIZATION,
    CONF_ENABLE_STT,
    CONF_STT_API_KEY,
    CONF_STT_API_VERSION,
    CONF_STT_BASE_URL,
    CONF_STT_LANGUAGE,
    CONF_STT_MODEL,
    CONF_STT_ORGANIZATION,
    DEFAULT_NAME,
    DEFAULT_STT_LANGUAGE,
    DEFAULT_STT_MODEL,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> bool:
    """Set up the STT entity when enabled in options."""
    options = {**entry.data, **entry.options}

    if not options.get(CONF_ENABLE_STT):
        _LOGGER.info("STT is disabled for entry %s; skipping STT entity setup", entry.entry_id)
        return True

    provider = ExtendedOpenAISttProvider(hass, entry)
    async_add_entities([ExtendedOpenAISttEntity(hass, provider, entry)])
    _LOGGER.info("Extended OpenAI STT entity added for entry %s", entry.entry_id)
    return True


class ExtendedOpenAISttEntity(SpeechToTextEntity):
    """Speech-to-Text entity for Extended OpenAI Conversation."""

    def __init__(self, hass: HomeAssistant, provider: Provider, entry: ConfigEntry) -> None:
        super().__init__()
        self.hass = hass
        self._provider = provider
        self._attr_unique_id = f"{entry.entry_id}_stt"
        # Use the config entry title to make the STT entity name specific per service/entry
        entry_title = getattr(entry, "title", None) or DEFAULT_NAME
        self._attr_name = f"{entry_title} STT"

    @property
    def provider(self) -> Provider:
        return self._provider

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        return await self._provider.async_process_audio_stream(metadata, stream)

    @property
    def supported_languages(self) -> list[str]:
        return self._provider.supported_languages

    @property
    def supported_formats(self) -> list[AudioFormats]:
        return self._provider.supported_formats

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        return self._provider.supported_codecs

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        return self._provider.supported_bit_rates

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        return self._provider.supported_sample_rates

    @property
    def supported_channels(self) -> list[AudioChannels]:
        return self._provider.supported_channels


class ExtendedOpenAISttProvider(Provider):
    """OpenAI-based STT provider."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self._entry = entry
        self._client: AsyncOpenAI | AsyncAzureOpenAI | None = None

        # Read options with fallbacks
        options = entry.options or {}
        data = entry.data or {}

        self._api_key: str | None = options.get(CONF_STT_API_KEY) or data.get(CONF_API_KEY)
        self._base_url: str | None = options.get(CONF_STT_BASE_URL) or data.get(CONF_BASE_URL)
        self._api_version: str | None = options.get(CONF_STT_API_VERSION) or data.get(CONF_API_VERSION)
        self._organization: str | None = options.get(CONF_STT_ORGANIZATION) or data.get(CONF_ORGANIZATION)
        # Extract model parameters using centralized function
        params = extract_and_validate_model_params(self._entry.options)
        self._model = params["model"]
        self._max_tokens = params["max_tokens"]
        self._temperature = params["temperature"]
        self._top_p = params["top_p"]
        self._language: str = options.get(CONF_STT_LANGUAGE, DEFAULT_STT_LANGUAGE)

        _LOGGER.debug(
            "Initialized OpenAI STT Provider (model=%s, base_url=%s, azure_api_version=%s)",
            self._model,
            self._base_url,
            self._api_version,
        )

    async def _ensure_client(self) -> None:
        if self._client is not None:
            return

        self._client = create_openai_client(
            hass=self.hass,
            api_key=self._api_key,
            base_url=self._base_url,
            api_version=self._api_version,
            organization=self._organization,
        )

    # Provider interface
    @property
    def default_language(self) -> str:
        # Use the first configured language as the default, preserving original form
        return (self._language or DEFAULT_STT_LANGUAGE).split(",")[0]

    @property
    def supported_languages(self) -> list[str]:
        # Accept both full locale (e.g., en-US) and base code (en)
        raw = (self._language or DEFAULT_STT_LANGUAGE)
        items = [s.strip() for s in raw.split(",") if s.strip()]
        expanded: list[str] = []
        for item in items:
            norm = item.replace("_", "-")
            if norm not in expanded:
                expanded.append(norm)
            # Add base code variant
            base = norm.split("-")[0].lower()
            if base and base not in expanded:
                expanded.append(base)
        return expanded

    @property
    def supported_formats(self) -> list[AudioFormats]:
        # Let Assist convert to WAV PCM for us
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        # Home Assistant renamed MONO -> CHANNEL_MONO in newer versions
        channel_mono = getattr(AudioChannels, "CHANNEL_MONO", None)
        if channel_mono is None:
            channel_mono = getattr(AudioChannels, "MONO", None)
        if channel_mono is None:
            try:
                # Fallback to enum value 1 if accessible
                channel_mono = AudioChannels(1)
            except Exception:  # noqa: BLE001
                pass
        return [channel_mono] if channel_mono is not None else []

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process an audio stream using OpenAI transcription API."""
        try:
            await self._ensure_client()

            # Collect stream into memory (Assist already chunks to short duration)
            audio_bytes = bytearray()
            async for chunk in stream:
                audio_bytes.extend(chunk)

            if not audio_bytes:
                _LOGGER.warning("Received empty audio stream")
                return SpeechResult("", SpeechResultState.ERROR)

            lang = (metadata.language or self._language or DEFAULT_STT_LANGUAGE)
            # Normalize: replace '_' with '-', reduce 'xx-YY' -> 'xx' for OpenAI
            if isinstance(lang, str):
                lang = lang.replace("_", "-")
                lang_param = lang.split("-")[0]
            else:
                lang_param = lang

            # Prepare file-like object for OpenAI client
            raw_bytes = bytes(audio_bytes)

            def _looks_like_wav(data: bytes) -> bool:
                return len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WAVE"

            if _looks_like_wav(raw_bytes):
                file_obj = BytesIO(raw_bytes)
                file_obj.name = "audio.wav"
            else:
                # Many HA pipelines stream raw PCM frames; wrap them into a valid WAV container
                wav_buf = BytesIO()
                channels = int(getattr(metadata, "channel", 1) or 1)
                sample_rate = int(getattr(metadata, "sample_rate", 16000) or 16000)
                sampwidth = 2  # 16-bit PCM
                with wave.open(wav_buf, "wb") as wf:
                    wf.setnchannels(channels)
                    wf.setsampwidth(sampwidth)
                    wf.setframerate(sample_rate)
                    wf.writeframes(raw_bytes)
                wav_buf.seek(0)
                wav_buf.name = "audio.wav"
                file_obj = wav_buf

            # Execute the async API call within the event loop
            transcription = await self._client.audio.transcriptions.create(
                model=self._model,
                file=file_obj,
                language=lang_param,
            )

            # Extract text from response
            text: str = getattr(transcription, "text", "")
            if text is None:
                text = ""

            result = SpeechResult(text, SpeechResultState.SUCCESS)
            # Optional metadata for downstream consumers
            result.metadata = {
                "model": self._model,
                "language": lang,
                "bytes": len(audio_bytes),
            }
            return result
        except OpenAIError as err:
            _LOGGER.error("OpenAI STT error: %s", err)
            res = SpeechResult("", SpeechResultState.ERROR)
            res.metadata = {"error": str(err)}
            return res
        except Exception as err:  # pylint: disable=broad-except
            _LOGGER.exception("Unexpected STT error")
            res = SpeechResult("", SpeechResultState.ERROR)
            res.metadata = {"error": str(err)}
            return res

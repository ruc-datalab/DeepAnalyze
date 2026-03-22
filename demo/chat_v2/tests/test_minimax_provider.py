"""Unit tests for MiniMax provider integration in chat service."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import sys
from pathlib import Path

# Add the backend_app parent to sys.path so we can import it.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from backend_app.services.chat import (
    MINIMAX_API_BASE,
    MINIMAX_DEFAULT_MODEL,
    ChatRuntimeConfig,
    _normalize_minimax_temperature,
    _normalize_temperature,
    build_chat_runtime_config,
)


# ---------------------------------------------------------------------------
# Temperature normalisation
# ---------------------------------------------------------------------------

class TestNormalizeMiniMaxTemperature:
    """Tests for the MiniMax-specific temperature clamping."""

    def test_clamp_zero_to_lower_bound(self):
        assert _normalize_minimax_temperature(0.0) == 0.01

    def test_clamp_negative_to_lower_bound(self):
        assert _normalize_minimax_temperature(-1.0) == 0.01

    def test_clamp_above_one(self):
        assert _normalize_minimax_temperature(1.5) == 1.0

    def test_value_within_range(self):
        assert _normalize_minimax_temperature(0.5) == 0.5

    def test_exact_lower_bound(self):
        assert _normalize_minimax_temperature(0.01) == 0.01

    def test_exact_upper_bound(self):
        assert _normalize_minimax_temperature(1.0) == 1.0

    def test_standard_temperature_unaffected(self):
        assert _normalize_minimax_temperature(0.4) == 0.4


# ---------------------------------------------------------------------------
# build_chat_runtime_config – MiniMax provider
# ---------------------------------------------------------------------------

class TestBuildChatRuntimeConfigMiniMax:
    """Tests for build_chat_runtime_config when provider is minimax."""

    def test_provider_recognised(self):
        cfg = build_chat_runtime_config({"provider": "minimax", "api_key": "key123"})
        assert cfg.provider == "minimax"

    def test_default_model(self):
        cfg = build_chat_runtime_config({"provider": "minimax", "api_key": "key123"})
        assert cfg.model == MINIMAX_DEFAULT_MODEL

    def test_default_api_base(self):
        cfg = build_chat_runtime_config({"provider": "minimax", "api_key": "key123"})
        assert cfg.api_base == MINIMAX_API_BASE

    def test_custom_api_base(self):
        cfg = build_chat_runtime_config({
            "provider": "minimax",
            "api_key": "key123",
            "api_base": "https://custom.api.com/v1",
        })
        assert cfg.api_base == "https://custom.api.com/v1"

    def test_custom_model(self):
        cfg = build_chat_runtime_config({
            "provider": "minimax",
            "api_key": "key123",
            "model": "MiniMax-M2.5-highspeed",
        })
        assert cfg.model == "MiniMax-M2.5-highspeed"

    def test_temperature_clamped_to_minimax_range(self):
        cfg = build_chat_runtime_config({
            "provider": "minimax",
            "api_key": "key123",
            "temperature": 1.8,
        })
        assert cfg.temperature == 1.0

    def test_zero_temperature_clamped(self):
        cfg = build_chat_runtime_config({
            "provider": "minimax",
            "api_key": "key123",
            "temperature": 0.0,
        })
        assert cfg.temperature == 0.01

    def test_api_key_stored(self):
        cfg = build_chat_runtime_config({"provider": "minimax", "api_key": "sk-test"})
        assert cfg.api_key == "sk-test"

    def test_unknown_provider_falls_back_to_local(self):
        cfg = build_chat_runtime_config({"provider": "unknown_provider"})
        assert cfg.provider == "local"

    def test_minimax_case_insensitive(self):
        cfg = build_chat_runtime_config({"provider": "MiniMax", "api_key": "k"})
        assert cfg.provider == "minimax"


# ---------------------------------------------------------------------------
# build_chat_runtime_config – existing providers still work
# ---------------------------------------------------------------------------

class TestBuildChatRuntimeConfigExistingProviders:
    """Regression tests: ensure local and heywhale providers are unaffected."""

    def test_local_provider_default(self):
        cfg = build_chat_runtime_config({})
        assert cfg.provider == "local"

    def test_heywhale_provider(self):
        cfg = build_chat_runtime_config({"provider": "heywhale", "api_key": "hw_key"})
        assert cfg.provider == "heywhale"
        assert "heywhale.com" in cfg.api_base

    def test_local_temperature_allows_full_range(self):
        cfg = build_chat_runtime_config({"provider": "local", "temperature": 1.8})
        assert cfg.temperature == 1.8


# ---------------------------------------------------------------------------
# _iter_minimax_stream – unit test with mocked OpenAI client
# ---------------------------------------------------------------------------

class TestIterMiniMaxStream:
    """Tests for _iter_minimax_stream using mocked OpenAI client."""

    def test_missing_api_key_raises(self):
        from backend_app.services.chat import _iter_minimax_stream

        cfg = ChatRuntimeConfig(provider="minimax", api_key="")
        with pytest.raises(ValueError, match="MiniMax API key is required"):
            list(_iter_minimax_stream([], cfg))

    @patch("backend_app.services.chat.openai.OpenAI")
    def test_yields_content_from_chunks(self, mock_openai_cls):
        from backend_app.services.chat import _iter_minimax_stream

        mock_delta = MagicMock()
        mock_delta.content = "Hello from MiniMax"

        mock_choice = MagicMock()
        mock_choice.delta = mock_delta

        mock_chunk = MagicMock()
        mock_chunk.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk])
        mock_openai_cls.return_value = mock_client

        cfg = ChatRuntimeConfig(
            provider="minimax",
            api_key="test-key",
            api_base=MINIMAX_API_BASE,
            model=MINIMAX_DEFAULT_MODEL,
            temperature=0.4,
        )
        results = list(_iter_minimax_stream([{"role": "user", "content": "hi"}], cfg))
        assert len(results) == 1
        assert results[0][0] == "Hello from MiniMax"

    @patch("backend_app.services.chat.openai.OpenAI")
    def test_passes_correct_params_to_openai_client(self, mock_openai_cls):
        from backend_app.services.chat import _iter_minimax_stream

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([])
        mock_openai_cls.return_value = mock_client

        cfg = ChatRuntimeConfig(
            provider="minimax",
            api_key="my-key",
            api_base="https://api.minimax.io/v1",
            model="MiniMax-M2.7",
            temperature=0.7,
        )
        conversation = [{"role": "user", "content": "test"}]
        list(_iter_minimax_stream(conversation, cfg))

        mock_openai_cls.assert_called_once_with(
            base_url="https://api.minimax.io/v1",
            api_key="my-key",
        )
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "MiniMax-M2.7"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["stream"] is True

    @patch("backend_app.services.chat.openai.OpenAI")
    def test_none_delta_yields_none(self, mock_openai_cls):
        from backend_app.services.chat import _iter_minimax_stream

        mock_chunk = MagicMock()
        mock_chunk.choices = []

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([mock_chunk])
        mock_openai_cls.return_value = mock_client

        cfg = ChatRuntimeConfig(
            provider="minimax",
            api_key="key",
            api_base=MINIMAX_API_BASE,
            model=MINIMAX_DEFAULT_MODEL,
            temperature=0.4,
        )
        results = list(_iter_minimax_stream([{"role": "user", "content": "hi"}], cfg))
        assert results[0][0] is None


# ---------------------------------------------------------------------------
# bot_stream provider routing
# ---------------------------------------------------------------------------

class TestProviderRouting:
    """Verify that the provider selection in build_chat_runtime_config routes correctly."""

    def test_minimax_config_creates_correct_provider(self):
        cfg = build_chat_runtime_config({
            "provider": "minimax",
            "api_key": "test-key",
        })
        assert cfg.provider == "minimax"
        assert cfg.api_base == MINIMAX_API_BASE
        assert cfg.model == MINIMAX_DEFAULT_MODEL

    def test_local_config_creates_correct_provider(self):
        cfg = build_chat_runtime_config({"provider": "local"})
        assert cfg.provider == "local"

    def test_heywhale_config_creates_correct_provider(self):
        cfg = build_chat_runtime_config({
            "provider": "heywhale",
            "api_key": "hw-key",
        })
        assert cfg.provider == "heywhale"

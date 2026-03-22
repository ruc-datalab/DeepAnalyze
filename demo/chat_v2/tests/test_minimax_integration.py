"""Integration tests for MiniMax provider.

These tests call the real MiniMax API and require a valid MINIMAX_API_KEY
environment variable. They are skipped by default in CI; run with:

    MINIMAX_API_KEY=<your-key> pytest -m integration demo/chat_v2/tests/
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_MINIMAX_KEY = os.environ.get("MINIMAX_API_KEY", "")
_skip_no_key = pytest.mark.skipif(
    not _MINIMAX_KEY,
    reason="MINIMAX_API_KEY not set – skipping integration tests",
)

from backend_app.services.chat import (
    MINIMAX_API_BASE,
    MINIMAX_DEFAULT_MODEL,
    ChatRuntimeConfig,
    _iter_minimax_stream,
    build_chat_runtime_config,
)


@_skip_no_key
@pytest.mark.integration
class TestMiniMaxLiveAPI:
    """Integration tests hitting the real MiniMax API."""

    def _make_config(self, **overrides) -> ChatRuntimeConfig:
        defaults = {
            "provider": "minimax",
            "api_key": _MINIMAX_KEY,
            "model": MINIMAX_DEFAULT_MODEL,
        }
        defaults.update(overrides)
        return build_chat_runtime_config(defaults)

    def test_basic_chat_completion(self):
        cfg = self._make_config()
        conversation = [{"role": "user", "content": "Say 'hello' and nothing else."}]
        chunks = list(_iter_minimax_stream(conversation, cfg))
        full_text = "".join(c for c, _ in chunks if c)
        assert len(full_text) > 0
        assert "hello" in full_text.lower()

    def test_streaming_returns_multiple_chunks(self):
        cfg = self._make_config()
        conversation = [{"role": "user", "content": "Count from 1 to 5, one number per line."}]
        chunks = list(_iter_minimax_stream(conversation, cfg))
        assert len(chunks) > 1

    def test_temperature_within_range(self):
        cfg = self._make_config(temperature=0.5)
        assert cfg.temperature == 0.5
        conversation = [{"role": "user", "content": "Say 'ok'."}]
        chunks = list(_iter_minimax_stream(conversation, cfg))
        full_text = "".join(c for c, _ in chunks if c)
        assert len(full_text) > 0

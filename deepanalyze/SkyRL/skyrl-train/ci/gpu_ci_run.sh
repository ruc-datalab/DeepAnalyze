#!/usr/bin/env bash
export CI=true
uv run --directory . --isolated --extra dev --extra vllm pytest -s tests/gpu/gpu_ci
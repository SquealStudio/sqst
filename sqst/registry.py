"""
Model registry and catalog for Squeal Studio models.
"""

from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import constants as hf_constants

from sqst.types import ModelInfo

# Full catalog of official Squeal Studio models
MODELS_CATALOG: Dict[str, ModelInfo] = {
    "Squeal-Studio/squeal_ai_8m-instruct": ModelInfo(
        repo_id="Squeal-Studio/squeal_ai_8m-instruct",
        name="squeal_ai_8m-instruct",
        aliases=["8m", "8m-instruct", "squeal-8m", "squeal-8m-instruct", "squeal_ai_8m", "squeal_ai_8m-instruct"],
        arch="qwen2",
        category="instruct",
        parameters="8.1M",
        context_length=1536,
        description="Compact ~8M Qwen2-based model fine-tuned to follow instructions in Russian.",
        trust_remote_code=False,
        has_onnx=False,
        default_gen_args={
            "max_new_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
    ),
    "Squeal-Studio/squeal_ai_8m-base": ModelInfo(
        repo_id="Squeal-Studio/squeal_ai_8m-base",
        name="squeal_ai_8m-base",
        aliases=["8m-base", "squeal-8m-base", "squeal_ai_8m-base"],
        arch="qwen2",
        category="base",
        parameters="8.1M",
        context_length=1536,
        description="Pretrained ~8M base language model (Qwen2) for Russian text completion.",
        trust_remote_code=False,
        has_onnx=False,
        default_gen_args={
            "max_new_tokens": 100,
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
            "do_sample": True,
        },
    ),
    "Squeal-Studio/squeal_ai_20m-instruct": ModelInfo(
        repo_id="Squeal-Studio/squeal_ai_20m-instruct",
        name="squeal_ai_20m-instruct",
        aliases=["20m", "20m-instruct", "squeal-20m", "squeal-20m-instruct", "squeal_ai_20m", "squeal_ai_20m-instruct"],
        arch="qwen2",
        category="instruct",
        parameters="20.3M",
        context_length=1024,
        description="Instruction-tuned ~20M model (Qwen2 + SFT on ru_turbo_alpaca). ONNX and quantized weights available.",
        trust_remote_code=False,
        has_onnx=True,
        default_gen_args={
            "max_new_tokens": 180,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
    ),
    "Squeal-Studio/squeal_ai_20m-base": ModelInfo(
        repo_id="Squeal-Studio/squeal_ai_20m-base",
        name="squeal_ai_20m-base",
        aliases=["20m-base", "squeal-20m-base", "squeal_ai_20m-base"],
        arch="qwen2",
        category="base",
        parameters="20.3M",
        context_length=1024,
        description="Pretrained ~20M base model (Qwen2) for text generation and continuation.",
        trust_remote_code=False,
        has_onnx=False,
        default_gen_args={
            "max_new_tokens": 120,
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
            "do_sample": True,
        },
    ),
    "Squeal-Studio/drakon_28m-instruct": ModelInfo(
        repo_id="Squeal-Studio/drakon_28m-instruct",
        name="drakon_28m-instruct",
        aliases=["drakon", "drakon-instruct", "drakon-28m", "drakon-28m-instruct", "drakon_28m-instruct"],
        arch="drakon",
        category="instruct",
        parameters="28.5M",
        context_length=1024,
        description="Hybrid ~28.5M Drakon model (Causal Conv + MinGRU + Causal Attention) trained on GrandMaster.",
        trust_remote_code=True,
        has_onnx=False,
        default_gen_args={
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
    ),
    "Squeal-Studio/drakon_28m-base": ModelInfo(
        repo_id="Squeal-Studio/drakon_28m-base",
        name="drakon_28m-base",
        aliases=["drakon-base", "drakon-28m-base", "drakon_28m-base"],
        arch="drakon",
        category="base",
        parameters="28.5M",
        context_length=1024,
        description="Base hybrid Drakon architecture, ~28.5M parameters (Depthwise Conv + MinGRU + Attention).",
        trust_remote_code=True,
        has_onnx=False,
        default_gen_args={
            "max_new_tokens": 150,
            "temperature": 0.8,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
    ),
    "Squeal-Studio/squeal_lstm_5m": ModelInfo(
        repo_id="Squeal-Studio/squeal_lstm_5m",
        name="squeal_lstm_5m",
        aliases=["lstm", "lstm-5m", "squeal-lstm", "squeal_lstm", "squeal_lstm_5m"],
        arch="lstm_lm",
        category="base",
        parameters="5.8M",
        context_length=256,
        description="Compact ~5.8M recurrent LSTM model for Russian text generation.",
        trust_remote_code=False,
        has_onnx=False,
        default_gen_args={
            "max_new_tokens": 100,
            "temperature": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "do_sample": True,
        },
    ),
}


def _build_alias_map() -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for repo, model_info in MODELS_CATALOG.items():
        alias_map[repo.lower()] = repo
        alias_map[model_info.name.lower()] = repo
        for alias in model_info.aliases:
            alias_map[alias.lower()] = repo
    return alias_map


# Alias lookup table
_ALIAS_MAP: Dict[str, str] = _build_alias_map()

# Prefixes that mark a string as a filesystem path rather than an alias / repo id.
_PATH_PREFIXES = ("./", "../", "/", "~", ".\\", "..\\")


def resolve_model_id(name_or_alias: str) -> str:
    """
    Resolves a friendly name or alias to a Hugging Face repo ID or local path.

    Strings that look like paths (``./model``, ``/abs/model``, ``~/model``) are
    returned unchanged, so a local directory can never be shadowed by an alias.
    Unknown names are returned as-is (allowing custom repos).

    Raises:
        ValueError: if the name is empty.
    """
    stripped = name_or_alias.strip()
    if not stripped:
        raise ValueError("Model name must not be empty.")

    if stripped.startswith(_PATH_PREFIXES):
        return stripped

    key = stripped.lower()
    if key in _ALIAS_MAP:
        return _ALIAS_MAP[key]

    # Try with prefix "squeal-studio/" if the user omitted it
    prefixed = f"squeal-studio/{key}"
    if prefixed in _ALIAS_MAP:
        return _ALIAS_MAP[prefixed]

    return stripped


def get_model_info(name_or_alias: str) -> Optional[ModelInfo]:
    """Returns the ModelInfo metadata if found in the catalog, else None."""
    resolved = resolve_model_id(name_or_alias)
    return MODELS_CATALOG.get(resolved)


def list_models() -> List[ModelInfo]:
    """Returns a list of all official models in the Squeal Studio catalog."""
    return list(MODELS_CATALOG.values())


# Files that indicate the weights are present (single-file or sharded checkpoints).
_WEIGHT_MARKERS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def is_cached_locally(repo_id: str) -> bool:
    """
    Checks whether the model's weights exist locally.

    For a local directory path, returns True if the directory exists. For a Hub
    repo, looks for single-file or sharded weights in the Hugging Face cache.
    """
    resolved = resolve_model_id(repo_id)
    if resolved.startswith(_PATH_PREFIXES):
        return Path(resolved).expanduser().is_dir()

    cache_dir = Path(hf_constants.HF_HUB_CACHE)
    if not cache_dir.exists():
        return False

    model_dir = cache_dir / ("models--" + resolved.replace("/", "--"))
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return False

    for snap in snapshots_dir.iterdir():
        if snap.is_dir() and any((snap / marker).exists() for marker in _WEIGHT_MARKERS):
            return True
    return False

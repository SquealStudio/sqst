"""
SQST: Lightweight Python inference library for Squeal Studio models from Hugging Face.
"""

from sqst.models.lstm import LSTMConfig, LSTMLanguageModel, register_lstm_architecture
from sqst.pipeline import SquealModel, chat, generate, load, stream
from sqst.registry import (
    MODELS_CATALOG,
    get_model_info,
    is_cached_locally,
    list_models,
    resolve_model_id,
)
from sqst.session import ChatSession
from sqst.types import (
    ArchitectureType,
    BackendType,
    ChatMessage,
    GenerationResult,
    ModelCategory,
    ModelInfo,
)

__version__ = "0.1.0"

__all__ = [
    # Main high-level API
    "load",
    "generate",
    "stream",
    "chat",
    "list_models",
    "get_model_info",
    "resolve_model_id",
    "is_cached_locally",
    # Core classes
    "SquealModel",
    "ChatSession",
    "ChatMessage",
    "ModelInfo",
    "GenerationResult",
    "ArchitectureType",
    "ModelCategory",
    "BackendType",
    "MODELS_CATALOG",
    # Architectures
    "LSTMConfig",
    "LSTMLanguageModel",
    "register_lstm_architecture",
    # Metadata
    "__version__",
]

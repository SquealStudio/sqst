"""
Data models and types for the sqst library.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

ArchitectureType = Literal["qwen2", "drakon", "lstm_lm"]
ModelCategory = Literal["instruct", "base"]
BackendType = Literal["pytorch", "onnx", "onnx-quantized"]


@dataclass
class ChatMessage:
    """A single message in a conversation."""
    role: Literal["user", "assistant", "system"]
    content: str


@dataclass
class GenerationResult:
    """Result of a model generation call."""
    text: str
    prompt: str
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: Optional[str] = None

    def __str__(self) -> str:
        return self.text


@dataclass
class ModelInfo:
    """Metadata describing a Squeal Studio model."""
    repo_id: str
    name: str
    aliases: List[str]
    arch: ArchitectureType
    category: ModelCategory
    parameters: str
    context_length: int
    description: str
    trust_remote_code: bool = False
    has_onnx: bool = False
    default_gen_args: Dict[str, Any] = field(default_factory=lambda: {
        "max_new_tokens": 150,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    })

    @property
    def is_instruct(self) -> bool:
        return self.category == "instruct"

    @property
    def is_base(self) -> bool:
        return self.category == "base"

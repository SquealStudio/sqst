"""
ONNX Runtime backend for models with exported ONNX graphs (e.g. squeal_ai_20m-instruct).
"""

from typing import Iterable, Iterator, List, Optional

import numpy as np
from huggingface_hub import hf_hub_download

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:  # pragma: no cover - depends on the environment
    ort = None  # type: ignore[assignment]
    HAS_ONNX = False

_RNG = np.random.default_rng()


def apply_repetition_penalty(logits: np.ndarray, token_ids: Iterable[int], penalty: float) -> np.ndarray:
    """
    Applies a repetition penalty in place, matching the Hugging Face semantics:
    positive logits are divided by ``penalty``, negative logits are multiplied by it,
    so repeated tokens always become *less* likely.
    """
    if penalty == 1.0:
        return logits
    idx = np.fromiter(set(int(t) for t in token_ids), dtype=np.int64)
    if idx.size == 0:
        return logits
    picked = logits[idx]
    logits[idx] = np.where(picked < 0, picked * penalty, picked / penalty)
    return logits


def sample_next_token(
    logits: np.ndarray,
    generated_ids: Iterable[int],
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Selects the next token id from a 1-D logits vector."""
    scores = np.array(logits, dtype=np.float64, copy=True)
    apply_repetition_penalty(scores, generated_ids, repetition_penalty)

    if not do_sample or temperature <= 0:
        return int(np.argmax(scores))

    scores /= temperature

    if 0 < top_k < scores.size:
        kth = np.partition(scores, -top_k)[-top_k]
        scores[scores < kth] = -np.inf

    scores -= np.max(scores)
    probs = np.exp(scores)
    probs /= probs.sum()

    if top_p is not None and 0.0 < top_p < 1.0:
        order = np.argsort(-probs)
        cumulative = np.cumsum(probs[order])
        keep = order[: int(np.searchsorted(cumulative, top_p)) + 1]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)
        probs /= probs.sum()

    return int((rng or _RNG).choice(probs.size, p=probs))


class ONNXModelWrapper:
    """Wrapper around an ONNX Runtime session for autoregressive text generation."""

    def __init__(
        self,
        repo_id: str,
        quantized: bool = False,
        provider: Optional[str] = None,
        use_gpu: bool = False,
        max_context: Optional[int] = None,
    ):
        """
        Args:
            repo_id: Hugging Face repo containing ``onnx/model.onnx`` (or the quantized file).
            quantized: Use ``onnx/model_quantized.onnx``.
            provider: Explicit ONNX Runtime execution provider (CPU is always kept as fallback).
            use_gpu: Prefer ``CUDAExecutionProvider`` when available and no provider is given.
            max_context: Sliding-window size in tokens; older tokens are dropped from the model input.
        """
        if not HAS_ONNX:
            raise ImportError(
                "onnxruntime is required to run the ONNX backend. "
                "Install it with: pip install onnxruntime (or pip install sqst[onnx])"
            )

        filename = "onnx/model_quantized.onnx" if quantized else "onnx/model.onnx"
        self.model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.max_context = max_context

        available = ort.get_available_providers()
        if provider:
            providers = [provider]
            if provider != "CPUExecutionProvider":
                providers.append("CPUExecutionProvider")
        else:
            providers = []
            if use_gpu and "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run_logits(self, input_ids: np.ndarray, attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Runs the model and returns logits for the sequence."""
        feed = {}
        if "input_ids" in self.input_names:
            feed["input_ids"] = input_ids.astype(np.int64)
        if "attention_mask" in self.input_names:
            if attention_mask is None:
                attention_mask = np.ones_like(input_ids, dtype=np.int64)
            feed["attention_mask"] = attention_mask.astype(np.int64)

        outputs = self.session.run(self.output_names, feed)
        return outputs[0]  # [batch, seq_len, vocab_size]

    def generate_iter(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        eos_token_id: int = 3,
    ) -> Iterator[int]:
        """Yields generated token ids one by one (greedy or sampling). EOS is not yielded."""
        curr_ids: List[int] = [int(t) for t in input_ids[0]]

        for _ in range(max_new_tokens):
            window = curr_ids[-self.max_context:] if self.max_context else curr_ids
            seq_array = np.array([window], dtype=np.int64)
            logits = self.run_logits(seq_array)[0, -1, :]

            next_id = sample_next_token(
                logits,
                curr_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
            )
            if next_id == eos_token_id:
                break

            curr_ids.append(next_id)
            yield next_id

    def generate(self, input_ids: np.ndarray, **kwargs) -> List[int]:
        """Runs autoregressive generation and returns all generated token ids."""
        return list(self.generate_iter(input_ids, **kwargs))

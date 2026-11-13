"""
Core inference pipeline and SquealModel wrapper.
"""

import threading
import warnings
from threading import Thread
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import transformers
from packaging.version import Version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from sqst.models.drakon import (
    clean_drakon_output,
    format_drakon_chat,
    format_drakon_prompt,
    make_drakon_stream_filter,
)
from sqst.models.lstm import register_lstm_architecture
from sqst.models.onnx_backend import HAS_ONNX, ONNXModelWrapper
from sqst.registry import get_model_info, resolve_model_id
from sqst.session import ChatSession
from sqst.textstream import StreamFilter, truncate_at_stop
from sqst.types import BackendType, ChatMessage, ModelInfo

# Ensure custom architectures are registered
register_lstm_architecture()

# `from_pretrained(dtype=...)` only exists in transformers >= 4.56; older releases use `torch_dtype`.
_DTYPE_KWARG = "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"

_VALID_BACKENDS = ("pytorch", "onnx", "onnx-quantized")

# Cache for singleton models, keyed by (resolved repo id, load options)
_MODEL_CACHE: Dict[Tuple[str, str], "SquealModel"] = {}
_CACHE_LOCK = threading.RLock()


def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    """Resolves 'auto', string, or torch.device into a valid torch.device."""
    if isinstance(device, torch.device):
        return device

    dev_str = str(device).lower().strip()
    if dev_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(dev_str)


def _resolve_dtype(dtype: Union[str, torch.dtype], device: torch.device) -> torch.dtype:
    """Selects an optimal torch dtype based on device."""
    if isinstance(dtype, torch.dtype):
        return dtype

    d_str = str(dtype).lower().strip()
    if d_str == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif device.type == "mps":
            return torch.float16
        else:
            return torch.float32

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if d_str not in mapping:
        raise ValueError(
            f"Unknown dtype {dtype!r}. Use 'auto' or one of: {', '.join(sorted(mapping))}."
        )
    return mapping[d_str]


def _merge_gen_args(
    info: Optional[ModelInfo],
    max_new_tokens: Optional[int],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    repetition_penalty: Optional[float],
    do_sample: Optional[bool],
) -> Dict[str, Any]:
    """Merges explicit arguments with the model's catalog defaults."""
    defaults = info.default_gen_args if info else {}

    def pick(value: Any, key: str, fallback: Any) -> Any:
        return value if value is not None else defaults.get(key, fallback)

    temp = pick(temperature, "temperature", 0.7)
    sample = bool(pick(do_sample, "do_sample", True))
    return {
        "max_new_tokens": pick(max_new_tokens, "max_new_tokens", 150),
        "temperature": temp,
        "top_p": pick(top_p, "top_p", 0.9),
        "top_k": pick(top_k, "top_k", 50),
        "repetition_penalty": pick(repetition_penalty, "repetition_penalty", 1.1),
        # Near-zero temperature means greedy decoding.
        "do_sample": sample and temp > 0.01,
    }


class _CancelFlag(StoppingCriteria):
    """Stopping criterion that lets the consumer cancel a background generation."""

    def __init__(self) -> None:
        self.stop = False

    def __call__(self, input_ids, scores, **kwargs):
        return torch.full((input_ids.shape[0],), self.stop, dtype=torch.bool, device=input_ids.device)


class _IncrementalDecoder:
    """Turns a stream of token ids into text pieces, holding back incomplete UTF-8 sequences."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.ids: List[int] = []
        self.printed = 0

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        if text.endswith("\ufffd"):  # partial multi-byte character, wait for more tokens
            return ""
        piece = text[self.printed:]
        self.printed = len(text)
        return piece

    def flush(self) -> str:
        text = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        piece = text[self.printed:]
        self.printed = len(text)
        return piece


class SquealModel:
    """
    Unified high-level interface for running Squeal Studio models.
    Supports Qwen2, Drakon, and LSTM architectures with PyTorch and ONNX backends.
    """

    def __init__(
        self,
        model_id: str,
        info: Optional[ModelInfo],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        model: Union[PreTrainedModel, ONNXModelWrapper],
        device: torch.device,
        dtype: torch.dtype,
        backend: BackendType = "pytorch",
    ):
        self.model_id = model_id
        self.info = info
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.dtype = dtype
        self.backend = backend

    # ------------------------------------------------------------ prompt format
    def format_prompt(self, prompt: str, raw: bool = False) -> str:
        """
        Formats a raw user prompt into the model's preferred input format.
        """
        if raw or not self.info:
            return prompt

        if self.info.arch == "drakon" and self.info.is_instruct:
            return format_drakon_prompt(prompt)

        if self.info.arch == "qwen2" and self.info.is_instruct:
            return f"<|user|>\n{prompt.strip()}\n<|assistant|>\n"

        return prompt

    def format_messages(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Formats a multi-turn conversation into a prompt string.

        Note: Drakon has no system-prompt slot, so ``system_prompt`` is ignored for it.
        """
        if self.info and self.info.arch == "drakon":
            return format_drakon_chat(messages)

        if self.info and self.info.arch == "qwen2" and self.info.is_instruct:
            formatted = ""
            if system_prompt:
                formatted += f"<|system|>\n{system_prompt.strip()}\n"
            for msg in messages:
                if msg.role == "user":
                    formatted += f"<|user|>\n{msg.content.strip()}\n"
                elif msg.role == "assistant":
                    formatted += f"<|assistant|>\n{msg.content.strip()}\n"
                elif msg.role == "system":
                    formatted += f"<|system|>\n{msg.content.strip()}\n"
            formatted += "<|assistant|>\n"
            return formatted

        # Default fallback for base models.
        # The Russian speaker labels are part of the plain-text dialogue format the
        # base models are prompted with; they are intentionally not translated.
        dialogue = ""
        if system_prompt:
            dialogue += f"{system_prompt.strip()}\n\n"
        for msg in messages:
            if msg.role == "system":
                dialogue += f"{msg.content.strip()}\n"
                continue
            speaker = "Пользователь" if msg.role == "user" else "Ассистент"
            dialogue += f"{speaker}: {msg.content.strip()}\n"
        dialogue += "Ассистент:"
        return dialogue

    def chat_stop_strings(self) -> List[str]:
        """
        Strings that mark the point where the model starts writing the *next* turn.
        Used by ChatSession so hallucinated follow-up turns never reach the history.
        """
        if self.info and self.info.arch == "drakon":
            return []  # handled by the Drakon stream filter / output cleaner
        if self.info and self.info.arch == "qwen2" and self.info.is_instruct:
            return ["<|user|>", "<|system|>", "<|assistant|>"]
        return ["\nПользователь:", "\nАссистент:"]

    # ------------------------------------------------------------ context budget
    @property
    def context_length(self) -> Optional[int]:
        return self.info.context_length if self.info else None

    def count_tokens(self, text: str) -> int:
        """Returns the number of tokens ``text`` occupies."""
        return len(self.tokenizer(text)["input_ids"])

    def prompt_token_budget(self, max_new_tokens: Optional[int] = None) -> Optional[int]:
        """
        Maximum number of prompt tokens that still leaves room for generation inside the
        model's context window (at least half of the window is always reserved for the prompt).
        Returns None when the context length is unknown.
        """
        ctx = self.context_length
        if ctx is None:
            return None
        if max_new_tokens is None:
            max_new_tokens = self.info.default_gen_args.get("max_new_tokens", 150)
        return max(ctx - max_new_tokens, ctx // 2)

    def _clip_new_tokens(self, params: Dict[str, Any], prompt_len: int) -> None:
        ctx = self.context_length
        if ctx is not None:
            params["max_new_tokens"] = max(1, min(params["max_new_tokens"], ctx - prompt_len))

    # ------------------------------------------------------------- generation
    @property
    def _is_onnx(self) -> bool:
        return self.backend in ("onnx", "onnx-quantized")

    @property
    def _uses_id_stream(self) -> bool:
        """True for backends that produce token ids one by one instead of via HF generate()."""
        if self._is_onnx:
            return True
        return bool(
            self.info and self.info.arch == "lstm_lm" and hasattr(self.model, "generate_recurrent")
        )

    def _warn_ignored(self, kwargs: Dict[str, Any]) -> None:
        if kwargs:
            warnings.warn(
                f"Ignoring generation kwargs not supported by this backend: {sorted(kwargs)}",
                stacklevel=3,
            )

    def _iter_token_ids(self, full_prompt: str, params: Dict[str, Any]) -> Iterator[int]:
        """Token-id generator for the ONNX backend and the recurrent LSTM fast path."""
        budget = self.prompt_token_budget(params["max_new_tokens"])

        if self._is_onnx:
            ids = self.tokenizer(full_prompt, return_tensors="np")["input_ids"]
            if budget is not None:
                ids = ids[:, -budget:]
            self._clip_new_tokens(params, ids.shape[1])
            eos = self.tokenizer.eos_token_id
            yield from self.model.generate_iter(
                input_ids=ids,
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_k=params["top_k"],
                top_p=params["top_p"],
                repetition_penalty=params["repetition_penalty"],
                do_sample=params["do_sample"],
                eos_token_id=eos if eos is not None else 3,
            )
            return

        ids = self.tokenizer(full_prompt, return_tensors="pt")["input_ids"]
        if budget is not None:
            ids = ids[:, -budget:]
        self._clip_new_tokens(params, ids.shape[1])
        yield from self.model.generate_recurrent(
            ids.to(self.device),
            max_new_tokens=params["max_new_tokens"],
            temperature=params["temperature"],
            top_k=params["top_k"],
            top_p=params["top_p"],
            repetition_penalty=params["repetition_penalty"],
            do_sample=params["do_sample"],
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def _encode(self, full_prompt: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenizes for the HF path, left-truncating to the context budget."""
        encoded = self.tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v for k, v in encoded.items() if k != "token_type_ids"}
        budget = self.prompt_token_budget(params["max_new_tokens"])
        if budget is not None and inputs["input_ids"].shape[1] > budget:
            inputs = {k: v[:, -budget:] for k, v in inputs.items()}
        self._clip_new_tokens(params, inputs["input_ids"].shape[1])
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _hf_gen_kwargs(
        self,
        params: Dict[str, Any],
        stop_strings: Optional[List[str]],
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": params["max_new_tokens"],
            "repetition_penalty": params["repetition_penalty"],
            "use_cache": not (self.info and self.info.arch in ("drakon", "lstm_lm")),
        }

        if params["do_sample"]:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = params["temperature"]
            if params["top_p"] is not None and params["top_p"] < 1.0:
                gen_kwargs["top_p"] = params["top_p"]
            if params["top_k"] is not None and params["top_k"] > 0:
                gen_kwargs["top_k"] = params["top_k"]
        else:
            gen_kwargs["do_sample"] = False

        if self.tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

        if stop_strings:  # lets HF stop early instead of generating to max_new_tokens
            gen_kwargs["stop_strings"] = list(stop_strings)
            gen_kwargs["tokenizer"] = self.tokenizer

        gen_kwargs.update(extra)
        return gen_kwargs

    def _postprocess(self, text: str, stop_strings: Optional[List[str]]) -> str:
        if self.info and self.info.arch == "drakon":
            text = clean_drakon_output(text)
        text = truncate_at_stop(text, stop_strings)
        return text.strip()

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        raw: bool = False,
        stop_strings: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generates text completion for the given prompt.

        Extra ``kwargs`` are forwarded to ``transformers`` ``generate()`` and are
        ignored (with a warning) by the ONNX and recurrent-LSTM backends.
        """
        full_prompt = self.format_prompt(prompt, raw=raw)
        params = _merge_gen_args(
            self.info, max_new_tokens, temperature, top_p, top_k, repetition_penalty, do_sample
        )

        if self._uses_id_stream:
            self._warn_ignored(kwargs)
            ids = list(self._iter_token_ids(full_prompt, params))
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            return self._postprocess(text, stop_strings)

        inputs = self._encode(full_prompt, params)
        prompt_len = inputs["input_ids"].shape[1]
        gen_kwargs = self._hf_gen_kwargs(params, stop_strings, kwargs)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        text = self.tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        return self._postprocess(text, stop_strings)

    def _raw_stream(
        self,
        full_prompt: str,
        params: Dict[str, Any],
        stop_strings: Optional[List[str]],
        kwargs: Dict[str, Any],
    ) -> Iterator[str]:
        """Yields unfiltered text pieces from whichever backend is in use."""
        if self._uses_id_stream:
            self._warn_ignored(kwargs)
            decoder = _IncrementalDecoder(self.tokenizer)
            for token_id in self._iter_token_ids(full_prompt, params):
                piece = decoder.push(token_id)
                if piece:
                    yield piece
            tail = decoder.flush()
            if tail:
                yield tail
            return

        inputs = self._encode(full_prompt, params)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        cancel = _CancelFlag()

        gen_kwargs = self._hf_gen_kwargs(params, stop_strings, kwargs)
        user_criteria = gen_kwargs.pop("stopping_criteria", None)
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([cancel] + list(user_criteria or []))
        gen_kwargs["streamer"] = streamer
        gen_kwargs.update(inputs)

        errors: List[BaseException] = []

        def _run() -> None:
            try:
                with torch.no_grad():  # grad mode is thread-local
                    self.model.generate(**gen_kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised in the consumer thread
                errors.append(exc)
                streamer.end()  # unblocks the consumer, otherwise it would wait forever

        thread = Thread(target=_run, daemon=True)
        thread.start()
        try:
            for piece in streamer:
                yield piece
        finally:
            cancel.stop = True  # stop generating if the consumer left early
            thread.join()
        if errors:
            raise errors[0]

    def stream(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        raw: bool = False,
        stop_strings: Optional[List[str]] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Streams generated text as it is produced.

        Works with every backend (PyTorch, ONNX, recurrent LSTM). Errors raised during
        generation are re-raised here, and abandoning the iterator cancels generation.
        """
        full_prompt = self.format_prompt(prompt, raw=raw)
        params = _merge_gen_args(
            self.info, max_new_tokens, temperature, top_p, top_k, repetition_penalty, do_sample
        )

        if self.info and self.info.arch == "drakon":
            flt = make_drakon_stream_filter(stop_strings)
        else:
            flt = StreamFilter(stops=stop_strings)

        source = self._raw_stream(full_prompt, params, stop_strings, kwargs)
        first = True
        try:
            for piece in source:
                out = flt.feed(piece)
                if first:
                    out = out.lstrip()  # mirrors the .strip() applied by generate()
                if out:
                    first = False
                    yield out
                if flt.done:
                    break
            tail = flt.flush()
            if first:
                tail = tail.lstrip()
            if tail:
                yield tail
        finally:
            source.close()

    def chat(self, system_prompt: Optional[str] = None) -> ChatSession:
        """
        Creates an interactive multi-turn ChatSession for this model.
        """
        return ChatSession(self, system_prompt=system_prompt)


def load(
    name_or_alias: str,
    device: Union[str, torch.device] = "auto",
    dtype: Union[str, torch.dtype] = "auto",
    backend: BackendType = "pytorch",
    trust_remote_code: Optional[bool] = None,
    **kwargs
) -> SquealModel:
    """
    Loads a Squeal Studio model by alias, repo ID, or local path.

    Args:
        trust_remote_code: Whether to execute code shipped with the model repo.
            Defaults to the catalog setting for official models and to ``False``
            for any other repo or path (you must opt in explicitly).
        **kwargs: Extra options forwarded to ``AutoModelForCausalLM.from_pretrained``.

    Examples:
        >>> model = sqst.load("8m")
        >>> model = sqst.load("20m", device="cuda")
        >>> model = sqst.load("20m", backend="onnx")
        >>> model = sqst.load("user/custom-repo", trust_remote_code=True)
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Use one of: {', '.join(_VALID_BACKENDS)}.")

    repo_id = resolve_model_id(name_or_alias)
    info = get_model_info(repo_id)

    target_device = _resolve_device(device)
    target_dtype = _resolve_dtype(dtype, target_device)

    if trust_remote_code is None:
        trust_remote = info.trust_remote_code if info else False
    else:
        trust_remote = trust_remote_code

    # Validate the backend before downloading anything.
    if backend in ("onnx", "onnx-quantized"):
        if not HAS_ONNX:
            raise ImportError(
                "onnxruntime is required for backend='onnx'. "
                "Install it with: pip install onnxruntime"
            )
        if info is not None and not info.has_onnx:
            raise ValueError(f"Model '{info.name}' has no ONNX weights; use backend='pytorch'.")

    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=trust_remote)
    # Set pad_token_id if missing
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 1. ONNX backend
    if backend in ("onnx", "onnx-quantized"):
        onnx_model = ONNXModelWrapper(
            repo_id=repo_id,
            quantized=(backend == "onnx-quantized"),
            use_gpu=(target_device.type == "cuda"),
            max_context=info.context_length if info else None,
        )
        return SquealModel(
            model_id=repo_id,
            info=info,
            tokenizer=tokenizer,
            model=onnx_model,
            device=target_device,
            dtype=target_dtype,
            backend=backend,
        )

    # 2. PyTorch backend
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote,
        _DTYPE_KWARG: target_dtype,
    }
    model_kwargs.update(kwargs)

    model = AutoModelForCausalLM.from_pretrained(repo_id, **model_kwargs)
    model.to(target_device)
    model.eval()

    return SquealModel(
        model_id=repo_id,
        info=info,
        tokenizer=tokenizer,
        model=model,
        device=target_device,
        dtype=target_dtype,
        backend="pytorch",
    )


def _get_cached_or_load(name_or_alias: str, load_kwargs: Optional[Dict[str, Any]] = None) -> SquealModel:
    """Returns a cached SquealModel for this name *and* load options, or loads a new one."""
    resolved = resolve_model_id(name_or_alias)
    options = load_kwargs or {}
    key = (resolved, repr(sorted((k, str(v)) for k, v in options.items())))
    with _CACHE_LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = load(resolved, **options)
        return _MODEL_CACHE[key]


def generate(
    model_name_or_alias: str,
    prompt: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    *,
    load_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> str:
    """
    Convenience function: one-line text generation with caching of the loaded model.

    Args:
        load_kwargs: Options for :func:`load` (``device``, ``dtype``, ``backend``, ...).
            Models are cached per distinct set of options.
        **kwargs: Generation options forwarded to :meth:`SquealModel.generate`.

    Example:
        >>> import sqst
        >>> print(sqst.generate("8m", "Привет, как твои дела?"))
    """
    model = _get_cached_or_load(model_name_or_alias, load_kwargs)
    return model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        **kwargs
    )


def stream(
    model_name_or_alias: str,
    prompt: str,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    *,
    load_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Iterator[str]:
    """
    Convenience function: one-line streaming generation with caching of the loaded model.

    Example:
        >>> import sqst
        >>> for token in sqst.stream("drakon", "Расскажи короткую историю"):
        ...     print(token, end="", flush=True)
    """
    model = _get_cached_or_load(model_name_or_alias, load_kwargs)
    yield from model.stream(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        **kwargs
    )


def chat(
    model_name_or_alias: str,
    system_prompt: Optional[str] = None,
    **load_kwargs
) -> ChatSession:
    """
    Convenience function: opens a ChatSession with the requested model.

    ``load_kwargs`` (``device``, ``dtype``, ``backend``, ...) are passed to :func:`load`.

    Example:
        >>> import sqst
        >>> conv = sqst.chat("20m")
        >>> print(conv.send("Привет!"))
    """
    model = _get_cached_or_load(model_name_or_alias, load_kwargs)
    return model.chat(system_prompt=system_prompt)

# Changelog

## 0.1.0

First public release.

### Features
- **Short aliases** for the official Squeal Studio models: `8m`, `8m-base`, `20m`, `20m-base`, `drakon`,
  `drakon-base`, `lstm`. Full repository names and local paths are also accepted.
- **Python API**: `sqst.load()`, `sqst.generate()`, `sqst.stream()` and `sqst.chat()` / `sqst.ChatSession`
  (`from sqst.session import ChatSession`).
- **Backends and architectures**: PyTorch (Qwen2), the hybrid Drakon architecture (Causal Conv + MinGRU +
  Causal Attention), the recurrent LSTM model (registered automatically in `AutoModelForCausalLM`), and
  ONNX Runtime (`backend="onnx"` / `"onnx-quantized"`, currently `20m` only).
- **Streaming** on every backend (PyTorch, ONNX, LSTM) with `stop_strings` support. Errors raised during
  generation are propagated to the caller, the worker thread is a daemon, and abandoning the iterator
  cancels generation.
- **Sampling controls**: `temperature`, `top_k`, `top_p`, `do_sample`, `repetition_penalty` (Hugging Face
  semantics, including negative logits), `max_new_tokens`, `stop_strings`.
- **Chat sessions** with the correct chat template per model (ChatML-style for Qwen2, GrandMaster for
  Drakon). History is trimmed (in the prompt only) to fit the model's context window, base models get stop
  strings so they cannot write the user's next turn, and a failed or interrupted turn leaves no dangling
  user message.
- **Model cache** keyed by load options (`device`, `dtype`, `backend`) and thread-safe. `generate()` and
  `stream()` take load options via `load_kwargs`.
- **Security**: `trust_remote_code` is enabled only for official catalog models that require it; for any
  other repository or path it defaults to `False` and must be requested explicitly with
  `load(..., trust_remote_code=True)` / `--trust-remote-code`.
- **CLI** (`sqst`): `list`, `info`, `run`, `chat`, `download`, with `--backend` and `--trust-remote-code`
  options. Chat commands: `/clear`, `/history`, `/undo`, `/exit`; `//text` sends a message that starts with
  a slash. Ctrl+C interrupts a reply without leaving the chat. Model and user text is never parsed as Rich
  markup.
- **Compatibility**: `load()` uses `dtype=` or `torch_dtype=` depending on the installed `transformers`
  version; unknown dtype names raise instead of silently falling back to float32.

### Notes
- The Russian strings that are part of the **models' prompt formats** (`Вопрос:` / `Ответ:`,
  `Пользователь:` / `Ассистент:`) and the Russian example prompts are intentional: translating them would
  break the models.
- Drakon's `<newline>` token is rendered as `"\n"` (an assumption about the token's meaning — adjust
  `DRAKON_TOKEN_REPLACEMENTS` if the tokenizer uses it differently).

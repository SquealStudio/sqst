# ⚡ SQST (Squeal Studio Inference Library)

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/sqst/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Squeal--Studio-yellow)](https://huggingface.co/Squeal-Studio)

A convenient, fast and lightweight Python library and CLI for running [Squeal Studio](https://huggingface.co/Squeal-Studio) language models.

> The models are trained for **Russian**, so the prompts in the examples below are in Russian.

---

## 🌟 Key features

- **Short aliases**: no need to type full repository names such as `Squeal-Studio/squeal_ai_8m-instruct` — just use `"8m"`, `"20m"`, `"drakon"` or `"lstm"`.
- **Works out of the box**:
  - Custom architectures are registered automatically (for example, `squeal_lstm_5m` ships no remote code on Hugging Face, but `sqst` registers the `LSTMLanguageModel` architecture in `AutoModelForCausalLM` natively).
  - Full support for the hybrid **Drakon** models (Causal Conv + MinGRU + Causal Attention), including special-token handling and the GrandMaster dataset format.
  - **ONNX Runtime** support for optimized inference (available for `squeal_ai_20m-instruct`).
- **One-line generation**: `sqst.generate("8m", "Привет!")`.
- **Streaming**: tokens are printed as soon as they are generated, on every backend (PyTorch, ONNX, LSTM).
- **Interactive chat (ChatSession)**: the conversation history is formatted with the right chat template (ChatML-style for Qwen2, GrandMaster for Drakon) and trimmed automatically to fit the model's context window.
- **Nice CLI**: a colorful interactive terminal with `/history`, `/undo` and `/clear` commands.

---

## 🤖 Supported models

| Alias | Full repository | Architecture | Parameters | Context | Purpose |
|---|---|---|---|---|---|
| `8m` / `8m-instruct` | `Squeal-Studio/squeal_ai_8m-instruct` | Qwen2 | 8.1M | 1536 | Instructions, Russian dialogue |
| `8m-base` | `Squeal-Studio/squeal_ai_8m-base` | Qwen2 | 8.1M | 1536 | Base text completion |
| `20m` / `20m-instruct` | `Squeal-Studio/squeal_ai_20m-instruct` | Qwen2 (ONNX) | 20.3M | 1024 | Instructions (ru_turbo_alpaca, ONNX support) |
| `20m-base` | `Squeal-Studio/squeal_ai_20m-base` | Qwen2 | 20.3M | 1024 | Base text completion |
| `drakon` | `Squeal-Studio/drakon_28m-instruct` | Drakon (minGRU+Attn) | 28.5M | 1024 | Hybrid architecture, question answering |
| `drakon-base` | `Squeal-Studio/drakon_28m-base` | Drakon (minGRU+Attn) | 28.5M | 1024 | Base hybrid model |
| `lstm` / `lstm-5m` | `Squeal-Studio/squeal_lstm_5m` | LSTM (recurrent) | 5.8M | 256 | Experimental ultra-light RNN |

---

## 📦 Installation

```bash
pip install sqst
```

For ONNX Runtime inference:
```bash
pip install "sqst[onnx]"
```

For local development:
```bash
git clone https://github.com/Squeal-Studio/sqst.git
cd sqst
pip install -e ".[dev]"
```

Requires Python 3.9 or newer.

---

## 🚀 Quick start (Python API)

### 1. One-line generation

```python
import sqst

# The model is downloaded on first use and cached for the lifetime of the process
response = sqst.generate("8m", "Привет, расскажи о себе!")
print(response)
```

### 2. Streaming generation

```python
import sqst

for chunk in sqst.stream("drakon", "Объясни, что такое фотосинтез."):
    print(chunk, end="", flush=True)
print()
```

### 3. Interactive chat

```python
import sqst

conversation = sqst.chat("20m")

# Turn 1
reply1 = conversation.send("Привет! Как тебя зовут?")
print("Bot:", reply1)

# Turn 2 (context is kept automatically)
reply2 = conversation.send("Что ты умеешь делать?")
print("Bot:", reply2)

# Streaming chat:
for chunk in conversation.stream("Расскажи шутку"):
    print(chunk, end="", flush=True)
```

If a generation fails or is interrupted before producing any text, the unanswered
user message is removed from the history again, so the session stays consistent.

### 4. Explicit loading and generation settings

```python
import sqst

# Device: "auto", "cuda", "cpu", "mps"
model = sqst.load("drakon", device="cuda", dtype="float16")

output = model.generate(
    "Придумай название для кофейни.",
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    repetition_penalty=1.1,
    stop_strings=["\n\n"],
)
print(output)
```

The one-line helpers take load options through `load_kwargs`, and models are cached
per distinct set of options:

```python
sqst.generate("20m", "Привет!", load_kwargs={"backend": "onnx", "device": "cpu"})
```

### 5. ONNX backend

```python
import sqst

model = sqst.load("20m", backend="onnx")            # or "onnx-quantized"
print(model.generate("Привет!", max_new_tokens=50))
```

Only models with ONNX weights support this backend (currently `20m`).

### 6. Custom repositories and `trust_remote_code`

For models **outside** the official catalog, `sqst` never executes code from the
repository unless you opt in explicitly:

```python
model = sqst.load("some-user/custom-model", trust_remote_code=True)
```

### 7. Using the recurrent `squeal_lstm_5m` model

`squeal_lstm_5m` ships no code in its HF repository, but `sqst` registers it automatically:

```python
import sqst

# Through the unified sqst API:
text = sqst.generate("lstm", "Однажды в лесу")
print(text)

# Or directly through transformers:
from transformers import AutoModelForCausalLM, AutoTokenizer
import sqst  # registers the LSTMLanguageModel architecture

tokenizer = AutoTokenizer.from_pretrained("Squeal-Studio/squeal_lstm_5m")
model = AutoModelForCausalLM.from_pretrained("Squeal-Studio/squeal_lstm_5m")
```

---

## 💻 Command-line tool (CLI)

After installation the `sqst` command is available:

### List available models
```bash
sqst list
```

### Show a model card
```bash
sqst info 20m
```

### Generate from the terminal
```bash
sqst run 8m "Напиши четверостишие о весне"
# with temperature and length settings:
sqst run drakon "Вопрос о науке" --temperature 0.6 --max-tokens 200
# ONNX backend:
sqst run 20m "Привет!" --backend onnx
```

### Interactive chat in the terminal
```bash
sqst chat 20m
```
Commands available inside the chat:
- `/clear` — clear the conversation history
- `/history` — show previous messages
- `/undo` — remove the last turn
- `/exit` — quit
- `//text` — send a message that starts with a slash

Press `Ctrl+C` during generation to interrupt the reply without leaving the chat.

### Pre-download weights
```bash
sqst download drakon
```

---

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

---

## 📄 License

Distributed under the [Apache 2.0](LICENSE) license.
Official website: [squealstudio.ru](https://squealstudio.ru)
Hugging Face: [huggingface.co/Squeal-Studio](https://huggingface.co/Squeal-Studio)

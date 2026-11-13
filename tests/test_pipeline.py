"""
Unit tests for pipeline utilities, prompt formatting, streaming and model loading.
"""

from unittest import mock

import numpy as np
import pytest
import torch

from sqst import pipeline
from sqst.pipeline import SquealModel, _merge_gen_args, _resolve_device, _resolve_dtype
from sqst.registry import get_model_info
from sqst.types import ChatMessage


def _model(alias, tokenizer=None, model=None, backend="pytorch"):
    info = get_model_info(alias)
    return SquealModel(
        model_id=info.repo_id,
        info=info,
        tokenizer=tokenizer,
        model=model,
        device=torch.device("cpu"),
        dtype=torch.float32,
        backend=backend,
    )


class FakeTokenizer:
    """Maps token ids to scripted text pieces; encodes one token per character."""

    eos_token_id = None
    pad_token_id = None

    def __init__(self, table=None):
        self.table = table or {i: chr(ord("a") + i) for i in range(26)}

    def __call__(self, text, return_tensors=None):
        ids = [ord(c) % 50 for c in text]
        if return_tensors == "np":
            return {"input_ids": np.array([ids], dtype=np.int64)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.table[int(i)] for i in ids)


class FakeIdModel:
    """Stands in for ONNXModelWrapper: yields scripted token ids."""

    def __init__(self, ids):
        self.ids = ids
        self.calls = []

    def generate_iter(self, **kwargs):
        self.calls.append(kwargs)
        yield from self.ids


# ----------------------------------------------------------------- utilities
def test_resolve_device():
    assert _resolve_device("cpu").type == "cpu"
    assert _resolve_device("auto").type in ("cuda", "mps", "cpu")


def test_resolve_dtype():
    cpu_dev = torch.device("cpu")
    assert _resolve_dtype("auto", cpu_dev) == torch.float32
    assert _resolve_dtype("float16", cpu_dev) == torch.float16


def test_resolve_dtype_rejects_unknown_names():
    with pytest.raises(ValueError):
        _resolve_dtype("float64", torch.device("cpu"))


def test_merge_gen_args_uses_catalog_defaults_and_greedy_switch():
    info = get_model_info("8m")
    params = _merge_gen_args(info, None, None, None, None, None, None)
    assert params["max_new_tokens"] == 150 and params["do_sample"] is True

    greedy = _merge_gen_args(info, None, 0.0, None, None, None, None)
    assert greedy["do_sample"] is False
    assert _merge_gen_args(info, None, 0.9, None, None, None, False)["do_sample"] is False


# ------------------------------------------------------------ prompt formats
def test_format_prompt_qwen2_instruct():
    model = _model("8m")
    assert model.format_prompt("Привет!") == "<|user|>\nПривет!\n<|assistant|>\n"
    assert model.format_prompt("Привет!", raw=True) == "Привет!"


def test_format_prompt_base_model():
    assert _model("8m-base").format_prompt("Текст начала") == "Текст начала"


def test_format_messages_qwen2_with_system_prompt():
    model = _model("8m")
    msgs = [ChatMessage("user", "hi"), ChatMessage("assistant", "yo"), ChatMessage("user", "ok")]
    out = model.format_messages(msgs, system_prompt="Be brief")
    assert out == (
        "<|system|>\nBe brief\n<|user|>\nhi\n<|assistant|>\nyo\n<|user|>\nok\n<|assistant|>\n"
    )


def test_format_messages_base_fallback_labels_roles():
    model = _model("8m-base")
    msgs = [ChatMessage("system", "rules"), ChatMessage("user", "hi")]
    out = model.format_messages(msgs)
    assert out == "rules\nПользователь: hi\nАссистент:"  # system is not mislabeled as the assistant


def test_chat_stop_strings():
    assert _model("drakon").chat_stop_strings() == []
    assert "<|user|>" in _model("8m").chat_stop_strings()
    assert "\nПользователь:" in _model("8m-base").chat_stop_strings()


# ------------------------------------------------------------- context budget
def test_prompt_token_budget():
    model = _model("lstm")  # context 256, default max_new_tokens 100
    assert model.prompt_token_budget() == 156
    assert model.prompt_token_budget(10) == 246
    assert model.prompt_token_budget(1000) == 128  # at least half the window stays for the prompt


def test_count_tokens_uses_tokenizer():
    assert _model("8m", tokenizer=FakeTokenizer()).count_tokens("abcd") == 4


# --------------------------------------------------------------- id streaming
def test_onnx_style_generate_and_stream_agree():
    ids = [0, 1, 2, 3]
    gen_model = _model("8m", FakeTokenizer(), FakeIdModel(ids), backend="onnx")
    text = gen_model.generate("hello", max_new_tokens=4)
    streamed = "".join(
        _model("8m", FakeTokenizer(), FakeIdModel(ids), backend="onnx").stream("hello", max_new_tokens=4)
    )
    assert text == streamed == "abcd"


def test_onnx_prompt_is_truncated_to_context_budget_and_max_tokens_clipped():
    fake = FakeIdModel([0])
    model = _model("lstm", FakeTokenizer(), fake, backend="onnx")  # context 256
    model.generate("x" * 1000, max_new_tokens=50, raw=True)
    call = fake.calls[0]
    assert call["input_ids"].shape[1] == 206      # 256 - 50
    assert call["max_new_tokens"] == 50


def test_generate_honours_stop_strings_on_id_backends():
    model = _model("8m-base", FakeTokenizer(), FakeIdModel([0, 1, 2, 3]), backend="onnx")
    assert model.generate("hi", stop_strings=["c"]) == "ab"


def test_stream_honours_stop_strings_split_across_tokens():
    table = {0: "x", 1: "ST", 2: "OP", 3: "y"}
    model = _model("8m-base", FakeTokenizer(table), FakeIdModel([0, 1, 2, 3]), backend="onnx")
    assert "".join(model.stream("hi", stop_strings=["STOP"])) == "x"


def test_drakon_stream_cuts_split_question_marker_and_cleans_tokens():
    table = {0: "Ответ", 1: " готов.", 2: "<newline>", 3: "\nВоп", 4: "рос:", 5: " дальше"}
    model = _model("drakon", FakeTokenizer(table), FakeIdModel([0, 1, 2, 3, 4, 5]), backend="onnx")
    out = "".join(model.stream("вопрос"))
    assert out.strip() == "Ответ готов."
    assert "Вопрос" not in out and "<newline>" not in out


def test_unsupported_kwargs_warn_on_id_backends():
    model = _model("8m", FakeTokenizer(), FakeIdModel([0]), backend="onnx")
    with pytest.warns(UserWarning):
        model.generate("hi", some_hf_only_option=1)


def test_abandoning_the_stream_closes_the_backend_generator():
    closed = []

    class Tracking(FakeIdModel):
        def generate_iter(self, **kwargs):
            try:
                yield from self.ids
            finally:
                closed.append(True)

    model = _model("8m", FakeTokenizer(), Tracking([0, 1, 2, 3]), backend="onnx")
    gen = model.stream("hi")
    next(gen)
    gen.close()
    assert closed == [True]


# ------------------------------------------------------------------- loading
def test_load_rejects_unknown_backend_before_downloading():
    with pytest.raises(ValueError):
        pipeline.load("8m", backend="tensorrt")


def _load_with_mocks(name, **kwargs):
    with mock.patch.object(pipeline.AutoTokenizer, "from_pretrained") as tok, \
         mock.patch.object(pipeline.AutoModelForCausalLM, "from_pretrained") as mdl:
        tok.return_value.pad_token_id = 0
        pipeline.load(name, device="cpu", dtype="float32", **kwargs)
        return tok.call_args.kwargs, mdl.call_args.kwargs


def test_trust_remote_code_defaults():
    tok_kw, mdl_kw = _load_with_mocks("custom-user/some-model")
    assert tok_kw["trust_remote_code"] is False and mdl_kw["trust_remote_code"] is False

    tok_kw, mdl_kw = _load_with_mocks("drakon")  # official model that needs remote code
    assert tok_kw["trust_remote_code"] is True and mdl_kw["trust_remote_code"] is True

    tok_kw, mdl_kw = _load_with_mocks("custom-user/some-model", trust_remote_code=True)
    assert mdl_kw["trust_remote_code"] is True


def test_dtype_keyword_matches_installed_transformers():
    _, mdl_kw = _load_with_mocks("8m")
    assert pipeline._DTYPE_KWARG in mdl_kw
    assert mdl_kw[pipeline._DTYPE_KWARG] == torch.float32


def test_model_cache_is_keyed_by_load_options():
    pipeline._MODEL_CACHE.clear()
    with mock.patch.object(pipeline, "load", side_effect=lambda name, **kw: object()) as fake_load:
        a = pipeline._get_cached_or_load("8m", {"device": "cpu"})
        b = pipeline._get_cached_or_load("squeal-8m", {"device": "cpu"})   # same model, same options
        c = pipeline._get_cached_or_load("8m", {"device": "cuda"})         # different options
        assert a is b and a is not c
        assert fake_load.call_count == 2
    pipeline._MODEL_CACHE.clear()

"""
Unit tests for LSTM model architecture and registration (require PyTorch).
"""

import torch

from sqst.models.lstm import (
    LSTMConfig,
    LSTMLanguageModel,
    _apply_repetition_penalty,
    _top_p_filter,
    register_lstm_architecture,
)


def _tiny_model(**overrides):
    cfg = dict(vocab_size=50, embed_dim=16, hidden_dim=32, num_layers=1, dropout=0.0)
    cfg.update(overrides)
    model = LSTMLanguageModel(LSTMConfig(**cfg))
    model.eval()
    return model


def test_lstm_config_and_registration():
    register_lstm_architecture()
    cfg = LSTMConfig(vocab_size=100, embed_dim=32, hidden_dim=64, num_layers=2)
    assert cfg.model_type == "lstm_lm"
    assert cfg.attribute_map == {"num_hidden_layers": "num_layers"}
    assert cfg.use_cache is False
    assert LSTMConfig(use_cache=True).use_cache is False  # cache protocol is unsupported


def test_lstm_model_forward():
    model = _tiny_model(vocab_size=100, embed_dim=32, hidden_dim=64, num_layers=2)
    input_ids = torch.randint(0, 100, (2, 5))

    output = model(input_ids)
    assert output.logits.shape == (2, 5, 100)
    assert output.loss is None

    labels = torch.randint(0, 100, (2, 5))
    output_with_loss = model(input_ids, labels=labels)
    assert output_with_loss.loss is not None
    assert output_with_loss.loss.item() > 0


def test_lstm_recurrent_generation():
    model = _tiny_model()
    prompt = torch.tensor([[1, 5, 10]])
    tokens = list(model.generate_recurrent(prompt, max_new_tokens=5, temperature=1.0, top_k=10))
    assert len(tokens) <= 5
    assert all(isinstance(t, int) for t in tokens)


def test_recurrent_greedy_is_deterministic_and_matches_full_forward():
    model = _tiny_model()
    prompt = torch.tensor([[1, 5, 10]])
    kwargs = dict(max_new_tokens=6, do_sample=False, repetition_penalty=1.0, eos_token_id=-1)
    first = list(model.generate_recurrent(prompt, **kwargs))
    assert first == list(model.generate_recurrent(prompt, **kwargs))

    # Same tokens as re-running the full sequence every step (no state carry-over).
    seq = prompt.clone()
    expected = []
    for _ in range(6):
        nxt = model(seq).logits[:, -1, :].argmax(-1, keepdim=True)
        expected.append(int(nxt))
        seq = torch.cat([seq, nxt], dim=1)
    assert first == expected


def test_recurrent_generation_restores_training_mode():
    model = _tiny_model()
    model.train()
    list(model.generate_recurrent(torch.tensor([[1, 2]]), max_new_tokens=2))
    assert model.training


def test_repetition_penalty_lowers_seen_tokens_of_either_sign():
    scores = torch.tensor([[2.0, -2.0, 1.0]])
    out = _apply_repetition_penalty(scores, {0, 1}, 1.5)
    assert out[0, 0] < 2.0 and out[0, 1] < -2.0 and out[0, 2] == 1.0
    assert scores[0, 1] == -2.0  # input is not modified


def test_top_p_filter_keeps_the_nucleus():
    scores = torch.tensor([[10.0, 9.0, -5.0, -5.0]])
    out = _top_p_filter(scores, 0.9)
    assert torch.isfinite(out[0, :2]).all()
    assert torch.isinf(out[0, 2:]).all()

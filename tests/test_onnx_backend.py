"""
Unit tests for the ONNX backend's sampling logic (no model download needed).
"""

import numpy as np

from sqst.models.onnx_backend import (
    ONNXModelWrapper,
    apply_repetition_penalty,
    sample_next_token,
)


def test_repetition_penalty_lowers_both_signs():
    logits = np.array([2.0, -2.0, 1.0], dtype=np.float64)
    apply_repetition_penalty(logits, [0, 1], 1.5)
    assert logits[0] < 2.0        # positive logit shrinks
    assert logits[1] < -2.0       # negative logit becomes MORE negative (less likely)
    assert logits[2] == 1.0       # untouched


def test_penalty_of_one_is_noop():
    logits = np.array([1.0, -1.0])
    apply_repetition_penalty(logits, [0, 1], 1.0)
    assert logits.tolist() == [1.0, -1.0]


def test_greedy_when_not_sampling():
    logits = np.array([0.1, 3.0, 0.2])
    assert sample_next_token(logits, [], do_sample=False, repetition_penalty=1.0) == 1


def test_top_k_one_is_deterministic():
    logits = np.array([0.1, 3.0, 0.2, 2.9])
    for _ in range(20):
        assert sample_next_token(logits, [], temperature=1.0, top_k=1, top_p=1.0,
                                 repetition_penalty=1.0) == 1


def test_top_p_keeps_only_the_nucleus():
    logits = np.array([10.0, 9.0, -5.0, -5.0])
    seen = {sample_next_token(logits, [], temperature=1.0, top_k=0, top_p=0.9,
                              repetition_penalty=1.0) for _ in range(200)}
    assert seen <= {0, 1}


class _FakeSession:
    """Always prefers token (last + 1) % vocab, so the output is predictable."""

    def __init__(self, vocab=6):
        self.vocab = vocab
        self.seen_lengths = []

    def run(self, output_names, feed):
        ids = feed["input_ids"]
        self.seen_lengths.append(ids.shape[1])
        logits = np.full((1, ids.shape[1], self.vocab), -5.0, dtype=np.float32)
        logits[0, -1, (int(ids[0, -1]) + 1) % self.vocab] = 5.0
        return [logits]


def _wrapper(max_context=None):
    w = object.__new__(ONNXModelWrapper)
    w.session = _FakeSession()
    w.input_names = ["input_ids"]
    w.output_names = ["logits"]
    w.max_context = max_context
    return w


def test_generate_iter_stops_at_eos_and_does_not_yield_it():
    w = _wrapper()
    ids = list(w.generate_iter(np.array([[0]]), max_new_tokens=10, do_sample=False,
                               repetition_penalty=1.0, eos_token_id=4))
    assert ids == [1, 2, 3]


def test_generate_returns_list():
    w = _wrapper()
    out = w.generate(np.array([[0]]), max_new_tokens=2, do_sample=False,
                     repetition_penalty=1.0, eos_token_id=99)
    assert out == [1, 2]


def test_sliding_window_limits_model_input():
    w = _wrapper(max_context=3)
    list(w.generate_iter(np.array([[0, 1, 2, 3, 4]]), max_new_tokens=4, do_sample=False,
                         repetition_penalty=1.0, eos_token_id=99))
    assert max(w.session.seen_lengths) == 3

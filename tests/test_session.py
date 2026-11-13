"""
Unit tests for ChatSession.
"""

import pytest

from sqst.session import ChatSession
from sqst.types import ChatMessage


class MockModel:
    model_id = "mock"

    def __init__(self, fail=False, chunks=("mock ", "reply")):
        self.fail = fail
        self.chunks = chunks
        self.prompts = []
        self.last_kwargs = {}

    def format_messages(self, messages, system_prompt=None):
        return "".join(f"<{m.role}>{m.content}" for m in messages)

    def generate(self, prompt, raw=True, **kwargs):
        self.prompts.append(prompt)
        self.last_kwargs = kwargs
        if self.fail:
            raise RuntimeError("boom")
        return "mock reply"

    def stream(self, prompt, raw=True, **kwargs):
        self.prompts.append(prompt)
        self.last_kwargs = kwargs
        for chunk in self.chunks:
            if self.fail:
                raise RuntimeError("boom")
            yield chunk


class BudgetModel(MockModel):
    """Counts one token per character and allows 40 prompt tokens."""

    def count_tokens(self, text):
        return len(text)

    def prompt_token_budget(self, max_new_tokens=None):
        return 40

    def chat_stop_strings(self):
        return ["<stop>"]


def test_chat_session_flow():
    session = ChatSession(MockModel())
    assert len(session) == 0

    assert session.send("Привет") == "mock reply"
    assert [m.role for m in session.history] == ["user", "assistant"]

    popped = session.undo()
    assert popped.content == "Привет"
    assert len(session) == 0

    session.send("Сообщение")
    assert len(session) == 2
    session.reset()
    assert len(session) == 0


def test_send_failure_rolls_back_user_message():
    session = ChatSession(MockModel(fail=True))
    with pytest.raises(RuntimeError):
        session.send("hi")
    assert len(session) == 0


def test_stream_collects_reply_into_history():
    session = ChatSession(MockModel())
    assert "".join(session.stream("hi")) == "mock reply"
    assert [m.content for m in session.history] == ["hi", "mock reply"]


def test_stream_failure_rolls_back():
    session = ChatSession(MockModel(fail=True))
    with pytest.raises(RuntimeError):
        list(session.stream("hi"))
    assert len(session) == 0


def test_stream_abandoned_early_keeps_partial_reply():
    session = ChatSession(MockModel())
    gen = session.stream("hi")
    assert next(gen) == "mock "
    gen.close()
    assert [m.content for m in session.history] == ["hi", "mock"]


def test_never_started_stream_leaves_history_untouched():
    session = ChatSession(MockModel())
    gen = session.stream("hi")
    gen.close()  # a generator that never started does not touch the history
    assert len(session) == 0


def test_old_turns_are_dropped_from_prompt_but_kept_in_history():
    model = BudgetModel()
    session = ChatSession(model)
    for i in range(4):
        session.history.append(ChatMessage(role="user", content=f"question number {i}"))
        session.history.append(ChatMessage(role="assistant", content=f"answer number {i}"))
    session.send("final")
    prompt = model.prompts[-1]
    assert len(prompt) <= 40 or prompt.count("<user>") == 1
    assert "final" in prompt
    assert "question number 0" not in prompt
    assert len(session) == 10  # nothing was deleted from the history


def test_default_stop_strings_are_supplied_but_can_be_overridden():
    model = BudgetModel()
    session = ChatSession(model)
    session.send("hi")
    assert model.last_kwargs["stop_strings"] == ["<stop>"]
    session.send("again", stop_strings=["X"])
    assert model.last_kwargs["stop_strings"] == ["X"]

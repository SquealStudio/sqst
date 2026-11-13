"""
Interactive chat sessions with conversation history and streaming.
"""

from typing import TYPE_CHECKING, Any, Generator, List, Optional
from sqst.types import ChatMessage

if TYPE_CHECKING:
    from sqst.pipeline import SquealModel


class ChatSession:
    """
    Manages a multi-turn conversation with a Squeal Studio model.

    - Applies the model's chat format to the history.
    - Drops the oldest turns from the *prompt* (not from ``history``) when the
      conversation would not fit into the model's context window.
    - Keeps ``history`` consistent: a failed or empty turn never leaves a
      dangling user message behind.
    """

    def __init__(self, model: "SquealModel", system_prompt: Optional[str] = None):
        self.model = model
        self.system_prompt = system_prompt
        self.history: List[ChatMessage] = []

    # ------------------------------------------------------------------ helpers
    def _prepare(self, kwargs: dict) -> str:
        """Builds the prompt for the current history and fills in default stop strings."""
        get_stops = getattr(self.model, "chat_stop_strings", None)
        if get_stops is not None and "stop_strings" not in kwargs:
            stops = get_stops()
            if stops:
                kwargs["stop_strings"] = stops

        messages = list(self.history)
        prompt = self.model.format_messages(messages, system_prompt=self.system_prompt)

        budget_fn = getattr(self.model, "prompt_token_budget", None)
        count_fn = getattr(self.model, "count_tokens", None)
        if budget_fn is None or count_fn is None:
            return prompt

        budget = budget_fn(kwargs.get("max_new_tokens"))
        if budget is None:
            return prompt

        while len(messages) > 1 and count_fn(prompt) > budget:
            messages = messages[1:]
            # Never start the prompt with an orphaned assistant reply.
            while len(messages) > 1 and messages[0].role == "assistant":
                messages = messages[1:]
            prompt = self.model.format_messages(messages, system_prompt=self.system_prompt)
        return prompt

    def _rollback(self, user_msg: ChatMessage) -> None:
        if self.history and self.history[-1] is user_msg:
            self.history.pop()

    # --------------------------------------------------------------------- API
    def send(self, message: str, **kwargs: Any) -> str:
        """Sends a user message and returns the model's reply."""
        user_msg = ChatMessage(role="user", content=message)
        self.history.append(user_msg)
        try:
            prompt = self._prepare(kwargs)
            reply = self.model.generate(prompt, raw=True, **kwargs)
        except BaseException:
            self._rollback(user_msg)
            raise

        self.history.append(ChatMessage(role="assistant", content=reply))
        return reply

    def stream(self, message: str, **kwargs: Any) -> Generator[str, None, None]:
        """
        Sends a user message and yields chunks of the reply in real time.

        The reply is added to the history when streaming finishes. If the consumer
        stops early, the partial reply is kept; if nothing was produced (or an
        error occurred) the user message is removed again.
        """
        user_msg = ChatMessage(role="user", content=message)
        self.history.append(user_msg)
        chunks: List[str] = []
        try:
            prompt = self._prepare(kwargs)
            for chunk in self.model.stream(prompt, raw=True, **kwargs):
                chunks.append(chunk)
                yield chunk
        except GeneratorExit:
            partial = "".join(chunks).strip()
            if partial:
                self.history.append(ChatMessage(role="assistant", content=partial))
            else:
                self._rollback(user_msg)
            raise
        except BaseException:
            self._rollback(user_msg)
            raise

        full_reply = "".join(chunks).strip()
        self.history.append(ChatMessage(role="assistant", content=full_reply))

    def reset(self) -> None:
        """Clears the chat history."""
        self.history.clear()

    def undo(self) -> Optional[ChatMessage]:
        """Removes the last turn (assistant and user messages) and returns the user message."""
        if not self.history:
            return None
        if self.history[-1].role == "assistant":
            self.history.pop()
        if self.history and self.history[-1].role == "user":
            return self.history.pop()
        return None

    def __len__(self) -> int:
        return len(self.history)

    def __repr__(self) -> str:
        return f"<ChatSession model={self.model.model_id} messages={len(self.history)}>"

"""
Helpers and prompt formatting for Drakon hybrid models.

NOTE: the Russian markers below ("Вопрос:" / "Ответ:") are the literal training
format of the Drakon models (GrandMaster dataset). They must NOT be translated.
"""

from typing import Dict, List, Optional

from sqst.textstream import StreamFilter
from sqst.types import ChatMessage

DRAKON_PROMPT_TEMPLATE = "Вопрос:\n{prompt}\n\nОтвет:\n"

# Text after which Drakon starts hallucinating the next question / ends the answer.
DRAKON_STOP_STRINGS: List[str] = ["<eos>", "\nВопрос:"]

# Special tokens that may leak into decoded text. "<newline>" encodes a line break.
DRAKON_TOKEN_REPLACEMENTS: Dict[str, str] = {
    "<newline>": "\n",
    "<bos>": "",
    "<pad>": "",
    "<unk>": "",
}


def format_drakon_prompt(prompt: str) -> str:
    """Formats a single question into Drakon's GrandMaster format."""
    return DRAKON_PROMPT_TEMPLATE.format(prompt=prompt.strip())


def format_drakon_chat(messages: List[ChatMessage]) -> str:
    """
    Formats a multi-turn conversation for Drakon instruct.

    Drakon has no system-prompt slot, so ``system`` messages are skipped.
    """
    formatted = ""
    for msg in messages:
        if msg.role == "user":
            formatted += DRAKON_PROMPT_TEMPLATE.format(prompt=msg.content.strip())
        elif msg.role == "assistant":
            formatted += f"{msg.content.strip()}<eos>\n"
    return formatted


def make_drakon_stream_filter(extra_stops: Optional[List[str]] = None) -> StreamFilter:
    """Creates a stream filter that cleans Drakon output chunk by chunk."""
    return StreamFilter(
        stops=DRAKON_STOP_STRINGS + list(extra_stops or []),
        replacements=DRAKON_TOKEN_REPLACEMENTS,
    )


def clean_drakon_output(text: str) -> str:
    """
    Strips special tokens and artifacts from Drakon outputs.

    The text is cut at ``<eos>`` or at a hallucinated follow-up question
    (``"\\nВопрос:"``). An answer that merely *contains* the word "Вопрос:"
    inline, or begins with it, is kept intact.
    """
    flt = make_drakon_stream_filter()
    return (flt.feed(text) + flt.flush()).strip()

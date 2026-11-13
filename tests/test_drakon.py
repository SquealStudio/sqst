"""
Unit tests for Drakon prompt formatting and output cleaning.

The Russian markers are the models' literal training format and are intentionally kept.
"""

from sqst.models.drakon import clean_drakon_output, format_drakon_chat, format_drakon_prompt
from sqst.types import ChatMessage


def test_drakon_format():
    formatted = format_drakon_prompt("Что такое фотосинтез?")
    assert formatted == "Вопрос:\nЧто такое фотосинтез?\n\nОтвет:\n"

    messages = [
        ChatMessage(role="user", content="Привет"),
        ChatMessage(role="assistant", content="Здравствуйте!"),
        ChatMessage(role="user", content="Как дела?"),
    ]
    chat_str = format_drakon_chat(messages)
    assert "Вопрос:\nПривет\n\nОтвет:\nЗдравствуйте!<eos>" in chat_str
    assert chat_str.endswith("Вопрос:\nКак дела?\n\nОтвет:\n")


def test_drakon_chat_skips_system_messages():
    messages = [
        ChatMessage(role="system", content="Be brief"),
        ChatMessage(role="user", content="Привет"),
    ]
    assert format_drakon_chat(messages) == "Вопрос:\nПривет\n\nОтвет:\n"


def test_clean_cuts_at_eos():
    assert clean_drakon_output("Привет! <eos>Вопрос: Что дальше?") == "Привет!"


def test_clean_cuts_hallucinated_next_question():
    assert clean_drakon_output("Ответ.\n\nВопрос: а дальше?") == "Ответ."


def test_clean_keeps_inline_word():
    text = "Ответ такой. Вопрос: как? Это важно."
    assert clean_drakon_output(text) == text


def test_clean_keeps_answer_starting_with_marker():
    assert clean_drakon_output("Вопрос: что это?") == "Вопрос: что это?"


def test_clean_newline_token_becomes_line_break():
    assert clean_drakon_output("строка1<newline>строка2") == "строка1\nстрока2"
    assert clean_drakon_output("<bos>текст<pad><unk>") == "текст"

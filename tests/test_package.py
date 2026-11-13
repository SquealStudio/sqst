"""
Package-level sanity tests.
"""

import re
from pathlib import Path

import sqst


def test_chat_is_the_convenience_function_and_session_module_is_reachable():
    assert callable(sqst.chat)
    assert not hasattr(sqst.chat, "ChatSession")   # `sqst.chat` is a function, not a module
    from sqst.session import ChatSession
    assert sqst.ChatSession is ChatSession


def test_version_matches_pyproject():
    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE).group(1)
    assert sqst.__version__ == declared

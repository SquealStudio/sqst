"""
Unit tests for the chunk-boundary-safe stream filter.
"""

from sqst.textstream import StreamFilter, truncate_at_stop


def _run(flt, chunks):
    out = []
    for c in chunks:
        out.append(flt.feed(c))
        if flt.done:
            break
    out.append(flt.flush())
    return "".join(out)


def test_stop_string_split_across_chunks():
    flt = StreamFilter(stops=["\nВопрос:"])
    text = _run(flt, ["Ответ ", "готов.\n\nВоп", "рос:", " что дальше?"])
    assert text == "Ответ готов.\n"
    assert flt.done


def test_partial_marker_is_released_when_it_never_completes():
    flt = StreamFilter(stops=["STOP"])
    assert _run(flt, ["abc ST", "ART"]) == "abc START"


def test_replacements_across_chunks():
    flt = StreamFilter(replacements={"<newline>": "\n", "<bos>": ""})
    assert _run(flt, ["a<new", "line>b<b", "os>c"]) == "a\nbc"


def test_earliest_marker_wins():
    flt = StreamFilter(stops=["<eos>"], replacements={"<newline>": "\n"})
    assert _run(flt, ["x<newline>y<eos>z"]) == "x\ny"


def test_truncate_at_stop():
    assert truncate_at_stop("hello\n\nworld", ["\n\n"]) == "hello"
    assert truncate_at_stop("hello", None) == "hello"
    assert truncate_at_stop("a-b-c", ["-c", "-b"]) == "a"

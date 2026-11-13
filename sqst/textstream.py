"""
Chunk-boundary-safe text filtering for streamed generation.

Streamed text arrives in arbitrary pieces, so a stop marker such as ``"\\nВопрос:"``
may be split across several chunks. :class:`StreamFilter` buffers just enough
text to detect such markers reliably.
"""

from typing import Dict, Iterable, Optional


def truncate_at_stop(text: str, stops: Optional[Iterable[str]]) -> str:
    """Cuts ``text`` at the earliest occurrence of any stop string."""
    cut = len(text)
    for stop in stops or []:
        if not stop:
            continue
        idx = text.find(stop)
        if idx != -1 and idx < cut:
            cut = idx
    return text[:cut]


class StreamFilter:
    """
    Incrementally cuts a text stream at stop strings and rewrites special tokens.

    Args:
        stops: Strings that terminate the stream (the stop string itself is not emitted).
        replacements: Mapping ``token -> replacement`` applied wherever the token occurs.
    """

    def __init__(
        self,
        stops: Optional[Iterable[str]] = None,
        replacements: Optional[Dict[str, str]] = None,
    ) -> None:
        self.stops = [s for s in (stops or []) if s]
        self.replacements = {k: v for k, v in (replacements or {}).items() if k}
        self.done = False
        self._buf = ""

    def _partial_suffix_len(self, text: str) -> int:
        """Length of the longest suffix of ``text`` that is a proper prefix of any marker."""
        best = 0
        for key in self.stops + list(self.replacements):
            for n in range(min(len(key) - 1, len(text)), best, -1):
                if text.endswith(key[:n]):
                    best = n
                    break
        return best

    def feed(self, chunk: str) -> str:
        """Consumes a chunk and returns the text that is safe to emit now."""
        if self.done or not chunk:
            return ""
        self._buf += chunk
        out = []
        while True:
            earliest = None  # (index, key, is_stop)
            for key in self.stops:
                idx = self._buf.find(key)
                if idx != -1 and (earliest is None or idx < earliest[0]):
                    earliest = (idx, key, True)
            for key in self.replacements:
                idx = self._buf.find(key)
                if idx != -1 and (earliest is None or idx < earliest[0]):
                    earliest = (idx, key, False)
            if earliest is None:
                break
            idx, key, is_stop = earliest
            out.append(self._buf[:idx])
            if is_stop:
                self._buf = ""
                self.done = True
                return "".join(out)
            out.append(self.replacements[key])
            self._buf = self._buf[idx + len(key):]

        hold = self._partial_suffix_len(self._buf)
        if hold:
            out.append(self._buf[:-hold])
            self._buf = self._buf[-hold:]
        else:
            out.append(self._buf)
            self._buf = ""
        return "".join(out)

    def flush(self) -> str:
        """Returns any text still held back once the stream has ended."""
        if self.done:
            return ""
        tail, self._buf = self._buf, ""
        return tail

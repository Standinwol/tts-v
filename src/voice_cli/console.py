from __future__ import annotations

import sys


def _reconfigure_streams() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_reconfigure_streams()

try:
    from rich.console import Console

    console = Console()
except ImportError:
    class _FallbackConsole:
        def print(self, *args, **kwargs) -> None:
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            print(*args, sep=sep, end=end)

    console = _FallbackConsole()

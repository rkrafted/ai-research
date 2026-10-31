"""``python3 -m bycloud <catalogue|captions|status|prompts>``."""

from __future__ import annotations

import sys

from . import ingest


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "prompts":
        from . import extract
        return extract.main(argv[1:])
    return ingest.main(argv)


if __name__ == "__main__":
    sys.exit(main())

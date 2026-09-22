"""Run versioned ChronoPDE V2 study administration commands."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from chronopde.v2.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

"""Dispatch versioned ChronoPDE V2 study administration commands."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from chronopde.v2.phase1 import main as phase1_main
from chronopde.v2.phase2 import main as phase2_main


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in {"phase1", "phase2"}:
        print("usage: chronopde_v2.py {phase1,phase2} [options]")
        return 4
    if arguments[0] == "phase1":
        return phase1_main(arguments)
    return phase2_main(arguments)

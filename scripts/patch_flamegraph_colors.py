"""Patch a pyinstrument HTML profile to use hot (yellow → orange → red) flame colors.

Usage
-----
    python scripts/patch_flamegraph_colors.py <input.html> [<output.html>]

If no output path is given the input file is patched in-place.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

# 11-stop warm palette: bright yellow → orange → deep red, mirroring the original length.
_HOT_PALETTE = (
    '["#FFE600","#FFB800","#FF8A00","#FF5C00","#FF2E00",'
    '"#FF0000","#CC0000","#FF0000","#FF2E00","#FF5C00","#FFE600"]'
)

# Matches any 11-element array of 6-digit hex colors immediately before .map(Ki)
_PALETTE_RE = re.compile(
    r'\["#[0-9A-Fa-f]{6}"(?:,"#[0-9A-Fa-f]{6}"){10}\](?=\.map\(Ki\))'
)


def patch(html: str) -> str:
    result, n = _PALETTE_RE.subn(_HOT_PALETTE, html)
    if n == 0:
        print(
            "warning: color palette not found — pyinstrument HTML structure may have changed.",
            file=sys.stderr,
        )
    return result


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)
    src = Path(args[0])
    dst = Path(args[1]) if len(args) > 1 else src
    dst.write_text(patch(src.read_text(encoding="utf-8")), encoding="utf-8")


if __name__ == "__main__":
    main()

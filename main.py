"""Compatibility wrapper for running BenchMFG from a source checkout."""

from pathlib import Path
import sys

src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from benchmfg.train import main  # noqa: E402

if __name__ == "__main__":
    main()

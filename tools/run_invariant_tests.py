#!/usr/bin/env python3
"""
Run just the gold-standard invariant tests.

Usage:
  python tools/run_invariant_tests.py

This runs:
  - tests/test_gold_standard_invariants.py
  - tests/test_gold_standard_pipeline_stub.py
"""

import sys
from pathlib import Path


def main() -> int:
    try:
        import pytest  # type: ignore
    except Exception:
        print("pytest is not installed. Install with: pip install pytest", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parent.parent
    tests = [
        repo_root / "tests" / "test_gold_standard_invariants.py",
        repo_root / "tests" / "test_gold_standard_pipeline_stub.py",
    ]
    args = ["-q"] + [str(p) for p in tests]
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())


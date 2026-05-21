"""Zero-dependency test runner.

pytest is not available in this environment, so this discovers ``test_*``
functions in every ``test_*.py`` module here and runs them with plain
assertions. Exit code 0 iff all pass. The test files are still written in a
pytest-compatible style (module-level ``test_*`` functions, plain ``assert``),
so ``pytest si_network_model/tests`` also works wherever pytest is installed.

Run from the repo root:  python -m si_network_model.tests.run_tests
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import traceback

_TESTS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent.parent


def main() -> int:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    test_files = sorted(_TESTS_DIR.glob("test_*.py"))
    total = passed = 0
    failures: list[str] = []

    for path in test_files:
        mod_name = f"si_network_model.tests.{path.stem}"
        module = importlib.import_module(mod_name)
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            fn = getattr(module, name)
            if not callable(fn):
                continue
            total += 1
            label = f"{path.stem}.{name}"
            try:
                fn()
            except Exception:  # noqa: BLE001 -- test runner reports everything
                failures.append(label)
                print(f"FAIL  {label}")
                traceback.print_exc()
            else:
                passed += 1
                print(f"pass  {label}")

    print(f"\n{passed}/{total} tests passed")
    if failures:
        print("failed: " + ", ".join(failures))
    return 0 if passed == total and total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

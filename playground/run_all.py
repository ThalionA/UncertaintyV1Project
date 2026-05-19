"""
Run all six similarity-framework figure scripts in order.

Usage:
    python run_all.py                  # writes to ./figures/
    SIMFRAME_OUTDIR=/path python run_all.py
"""
import subprocess
import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "fig01.py", "fig02.py", "fig03.py",
    "fig04.py", "fig05.py", "fig06.py",
    "fig07.py", "fig08.py", "fig09.py",
    "fig10.py", "fig11.py", "fig12.py",
    "fig13.py", "fig14.py",
    "fig15.py", "fig16.py",
]


def main():
    for name in SCRIPTS:
        path = os.path.join(_THIS_DIR, name)
        print(f"--- running {name} ---")
        result = subprocess.run([sys.executable, path], cwd=_THIS_DIR)
        if result.returncode != 0:
            print(f"!! {name} failed with code {result.returncode}")
            sys.exit(result.returncode)
    print("\nAll six figures generated.")
    outdir = os.environ.get("SIMFRAME_OUTDIR",
                            os.path.join(_THIS_DIR, "figures"))
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()

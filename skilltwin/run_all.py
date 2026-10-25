"""Run the whole Demo A pipeline (steps 2-10) from the raw dataset folder.

Usage (from the dataset root):  python -m skilltwin.run_all [--llm MODEL]
"""

import argparse
import sys
import time

from skilltwin import charts, classify, coach, expert, features, preprocess, score
from skilltwin.adapters import mendeley

STEPS = [("2 loader", mendeley), ("3 clean", preprocess), ("4-5 features", features),
         ("6 expert band", expert), ("7 score", score), ("8 classifier", classify),
         ("9 coach", coach), ("10 charts", charts)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default=None)
    args = ap.parse_args()
    for name, module in STEPS:
        t = time.time()
        print(f"\n===== step {name} =====", flush=True)
        sys.argv = [module.__name__] + (["--llm", args.llm] if module is coach and args.llm else [])
        module.main()
        print(f"[{name}: {time.time() - t:.0f}s]")


if __name__ == "__main__":
    main()

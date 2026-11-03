#!/usr/bin/env python
"""Formal end-to-end LLM experiment orchestrator.

Stops only on hard protocol failures listed in the experiment plan.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def run(cmd: list[str], check: bool = True) -> int:
    print("\n>>>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if check and proc.returncode != 0:
        print(f"STOP: command failed rc={proc.returncode}: {' '.join(cmd)}")
        sys.exit(proc.returncode)
    return proc.returncode


def main() -> None:
    py = sys.executable

    # 1) frozen dataset hash
    run([py, "scripts/validate_frozen_dataset.py"])

    # 2) pytest
    run([py, "-m", "pytest", "tests", "-q"])

    # 2b) Exp2 artifacts (QA frozen)
    run([py, "scripts/build_exp2_artifacts.py"])

    # 3) API connectivity
    zhipu_ok = run([py, "scripts/check_api.py", "--provider", "zhipu"], check=False) == 0
    bailian_ok = run([py, "scripts/check_api.py", "--provider", "bailian"], check=False) == 0
    if not zhipu_ok and not bailian_ok:
        print("STOP: both providers unavailable")
        sys.exit(10)
    providers = []
    if zhipu_ok:
        providers.append("zhipu")
    if bailian_ok:
        providers.append("bailian")
    print(f"Available providers: {providers}")

    # 4) dual smoke
    smoke_rc = run(
        [py, "scripts/run_smoke.py", "--providers", ",".join(providers)],
        check=False,
    )
    if smoke_rc == 5:
        print("STOP: COST_GUARD_TRIGGERED")
        sys.exit(5)
    if smoke_rc != 0:
        print("STOP: smoke sanity failed")
        sys.exit(smoke_rc)

    # 5) Exp1
    for p in providers:
        run([py, "scripts/run_exp1.py", "--provider", p])

    # 6) Exp2A
    for p in providers:
        run([py, "scripts/run_exp2_overdeletion.py", "--provider", p])

    # 7) Exp2B
    for p in providers:
        run([py, "scripts/run_exp2_residual.py", "--provider", p])

    # 8) summarize
    run([py, "scripts/summarize_results.py"])
    print("\nFORMAL_PIPELINE_COMPLETE")
    print(f"Report: {config.FINAL_DIR / 'LLM_FINAL_REPORT.md'}")


if __name__ == "__main__":
    main()

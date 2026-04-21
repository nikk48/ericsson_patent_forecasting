"""
run_all_tasks.py

Run Task 1, Task 2, and Task 3 sequentially.

This script is useful for a fully reproducible coursework run:
1) Task 1 EDA
2) Task 2 baseline forecasting
3) Task 3 segmentation-based forecasting
"""

from pathlib import Path
import subprocess
import sys
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_step(step_name: str, script_name: str) -> None:
    """Run one driver script and raise on failure."""
    script_path = PROJECT_ROOT / "drivers" / script_name
    print(f"\n===== {step_name} =====")
    mpl_dir = PROJECT_ROOT / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(mpl_dir))
    env.setdefault("MPLBACKEND", "Agg")
    if not env.get("LOKY_MAX_CPU_COUNT"):
        env["LOKY_MAX_CPU_COUNT"] = str(min(8, os.cpu_count() or 1))
    subprocess.run([sys.executable, str(script_path)], check=True, env=env)


def main() -> None:
    """Run the full pipeline."""
    print("Starting full coursework pipeline (Task 1 -> Task 2 -> Task 3)...")
    _run_step("TASK 1: EDA", "run_eda.py")
    _run_step("TASK 2: BASELINE FORECASTING", "run_forecasting.py")
    _run_step("TASK 3: SEGMENTATION FORECASTING", "run_task3.py")
    print("\nAll tasks completed successfully.")
    print(f"Outputs available in: {PROJECT_ROOT / 'outputs'}")


if __name__ == "__main__":
    main()

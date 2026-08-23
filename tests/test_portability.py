"""Regression tests for repository-relative command-line entry points."""
from pathlib import Path
import subprocess
import sys

from src import final_validation, ml_experiment, real_data, robustness, run_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_evaluation_paths_are_repository_relative():
    assert run_evaluation.DATA_CSV == PROJECT_ROOT / "data" / "dataset_42d.csv"
    assert run_evaluation.RESULT_DIR == PROJECT_ROOT / "results"
    assert final_validation.RESULT_DIR == PROJECT_ROOT / "results"
    assert robustness.RESULT_DIR == PROJECT_ROOT / "results"
    assert ml_experiment.RESULT_DIR == PROJECT_ROOT / "results"
    assert real_data.KDD == PROJECT_ROOT / "data" / "real" / "KDDTrain+.txt"


def test_data_generator_runs_outside_repository(tmp_path):
    script = PROJECT_ROOT / "src" / "data_gen.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Phase 1 validation: OK" in completed.stdout
    assert (PROJECT_ROOT / "data" / "dataset_42d.csv").exists()

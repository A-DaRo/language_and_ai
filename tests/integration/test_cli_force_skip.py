from click.testing import CliRunner
from neuro_stylometry.__main__ import cli
from pathlib import Path
import tempfile


def test_cli_dry_run_injects_force_skip(tmp_path):
    runner = CliRunner()
    # Use an existing file path (README.md) to satisfy 'dataset exists' check
    repo_root = Path(__file__).resolve().parents[2]
    dataset = repo_root / "README.md"
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    result = runner.invoke(
        cli,
        [
            "run-phase-a",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--force-skip",
        ],
    )
    assert result.exit_code == 0
    # The resolved configuration dump should contain execution.force_skip: true
    assert "force_skip" in result.output
    assert "true" in result.output.lower() or "True" in result.output

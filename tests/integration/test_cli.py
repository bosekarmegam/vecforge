# VecForge — Tests for CLI
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from vecforge.cli.main import cli
from vecforge.core.vault import VecForge


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def test_vault(tmp_path):
    vault_path = str(tmp_path / "cli_test.db")
    db = VecForge(vault_path)
    db.add("CLI test document", namespace="default")
    db.close()
    return vault_path


def test_cli_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "VecForge" in result.output


def test_cli_ingest(runner, tmp_path):
    vault_path = str(tmp_path / "ingest.db")
    doc_path = tmp_path / "test.txt"
    doc_path.write_text("Hello VecForge CLI.")

    result = runner.invoke(cli, ["ingest", str(tmp_path), "--vault", vault_path])
    assert result.exit_code == 0
    assert "Ingested" in result.output


def test_cli_search(runner, test_vault):
    result = runner.invoke(cli, ["search", "test document", "--vault", test_vault])
    assert result.exit_code == 0
    assert "Result 1" in result.output
    assert "Score: " in result.output

    # Search with no results (assuming high threshold or nothing related)
    # VecForge might return something due to BM25, but we test the invocation.
    result_empty = runner.invoke(
        cli, ["search", "completelyunrelatedgibberish", "--vault", test_vault]
    )
    assert result_empty.exit_code == 0


def test_cli_stats(runner, test_vault):
    result = runner.invoke(cli, ["stats", test_vault])
    assert result.exit_code == 0
    assert "VecForge Vault Statistics" in result.output
    assert "Documents:" in result.output


def test_cli_export(runner, test_vault, tmp_path):
    out_file = tmp_path / "export.json"
    result = runner.invoke(cli, ["export", test_vault, "-o", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()
    assert "Exported 1 documents" in result.output

    # Export to stdout
    result = runner.invoke(cli, ["export", test_vault])
    assert result.exit_code == 0
    assert "CLI test document" in result.output


@patch("uvicorn.run")
def test_cli_serve(mock_run, runner, test_vault):
    result = runner.invoke(cli, ["serve", "--vault", test_vault, "--port", "8080"])
    assert result.exit_code == 0
    mock_run.assert_called_once()
    assert "VecForge REST Server" in result.output

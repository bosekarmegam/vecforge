# VecForge — Tests for the 2.0 CLI (hash embedder, no model download)
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from vecforge import __version__
from vecforge.cli.main import cli
from vecforge.core.vault import VecForge


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def vault_path(tmp_path: Path) -> str:
    path = str(tmp_path / "cli.db")
    db = VecForge(path, model_name="hash")
    db.add("patient with diabetes", metadata={"year": 2024})
    db.add("hip fracture", metadata={"year": 2020})
    db.close()
    return path


def test_version_comes_from_package(runner: CliRunner) -> None:
    out = runner.invoke(cli, ["--version"]).output
    assert __version__ in out


def test_search_json_quantum_filters(runner: CliRunner, vault_path: str) -> None:
    res = runner.invoke(
        cli,
        [
            "search",
            "diabetic patient",
            "--vault",
            vault_path,
            "--model",
            "hash",
            "--quantum",
            "--json",
            "--filters",
            '{"year": {"gte": 2023}}',
        ],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert len(data) == 1 and data[0]["text"] == "patient with diabetes"
    bad = runner.invoke(
        cli,
        [
            "search",
            "x",
            "--vault",
            vault_path,
            "--model",
            "hash",
            "--filters",
            "notjson",
        ],
    )
    assert bad.exit_code != 0


def test_stats_compact_backup(
    runner: CliRunner, vault_path: str, tmp_path: Path
) -> None:
    res = runner.invoke(cli, ["stats", vault_path, "--model", "hash"])
    assert res.exit_code == 0 and "Embedder:       hash:384" in res.output
    res = runner.invoke(cli, ["compact", vault_path, "--model", "hash"])
    assert res.exit_code == 0 and "Compacted" in res.output
    dest = str(tmp_path / "copy.db")
    res = runner.invoke(cli, ["backup", vault_path, dest])
    assert res.exit_code == 0 and Path(dest).exists()


def test_export_rejects_other_formats(runner: CliRunner, vault_path: str) -> None:
    assert runner.invoke(cli, ["export", vault_path, "--format", "csv"]).exit_code != 0


def test_memory_commands(runner: CliRunner, tmp_path: Path) -> None:
    mem = str(tmp_path / "mem.db")
    base = ["--vault", mem, "--agent", "cli", "--model", "hash"]
    res = runner.invoke(
        cli,
        [
            "memory",
            "remember",
            "User likes tea",
            "--kind",
            "preference",
            "--tags",
            "drinks,",
        ]
        + base,
    )
    assert res.exit_code == 0 and "Remembered" in res.output
    res = runner.invoke(
        cli, ["memory", "remember", "User likes tea", "--kind", "preference"] + base
    )
    assert "Reinforced" in res.output
    res = runner.invoke(
        cli, ["memory", "recall", "beverage preference", "--json"] + base
    )
    assert (
        res.exit_code == 0 and json.loads(res.output)[0]["content"] == "User likes tea"
    )
    res = runner.invoke(cli, ["memory", "list"] + base)
    assert "User likes tea" in res.output
    res = runner.invoke(cli, ["memory", "context", "--query", "tea"] + base)
    assert "Preferences" in res.output
    out = str(tmp_path / "mem.json")
    assert runner.invoke(cli, ["memory", "export", out] + base).exit_code == 0
    res = runner.invoke(
        cli,
        [
            "memory",
            "import",
            out,
            "--vault",
            str(tmp_path / "mem2.db"),
            "--agent",
            "cli",
            "--model",
            "hash",
        ],
    )
    assert "Imported 1" in res.output
    mid = json.loads(
        runner.invoke(cli, ["memory", "recall", "tea", "--json"] + base).output
    )[0]["memory_id"]
    assert runner.invoke(cli, ["memory", "forget", mid] + base).exit_code == 0
    assert runner.invoke(cli, ["memory", "forget", mid] + base).exit_code == 1
    assert "No sessions" in runner.invoke(cli, ["memory", "sessions"] + base).output


def test_mcp_print_config(runner: CliRunner) -> None:
    res = runner.invoke(
        cli, ["mcp", "--print-config", "claude-desktop", "--vault", "/x.db"]
    )
    assert res.exit_code == 0
    cfg = json.loads(res.output)
    assert "vecforge-memory" in cfg["mcpServers"]


@patch("vecforge.mcp.server.run_server")
def test_mcp_starts_server(mock_run, runner: CliRunner) -> None:
    res = runner.invoke(
        cli,
        [
            "mcp",
            "--vault",
            "/tmp/x.db",
            "--agent",
            "a",
            "--transport",
            "http",
            "--port",
            "9999",
        ],
    )
    assert res.exit_code == 0, res.output
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["transport"] == "http"
    assert mock_run.call_args.kwargs["port"] == 9999


@patch("uvicorn.run")
def test_serve_passes_api_key(mock_run, runner: CliRunner, vault_path: str) -> None:
    res = runner.invoke(
        cli, ["serve", "--vault", vault_path, "--model", "hash", "--api-key", "k"]
    )
    assert res.exit_code == 0 and "API key required" in res.output
    assert mock_run.call_args.kwargs["host"] == "127.0.0.1"

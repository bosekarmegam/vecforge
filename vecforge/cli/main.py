# VecForge — Universal Local-First Vector Database
# Copyright (c) 2026 Suneel Bose K · ArcGX TechLabs Private Limited
# Built by Suneel Bose K (Founder & CEO, ArcGX TechLabs)
#
# Licensed under the Business Source License 1.1 (BSL 1.1)
# Free for personal, research, open-source, and non-commercial use.
# Commercial use requires a separate license from ArcGX TechLabs.
# See LICENSE file in the project root or contact: suneelbose@arcgx.in

"""
VecForge CLI — command-line interface.

Commands: ingest, search, stats, export, serve, compact, backup, mcp and
the ``memory`` group (remember / recall / list / sessions / context /
export / import / forget).

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import click

from vecforge import __version__

_MODEL_HELP = "Embedding model name, or 'hash' for the dependency-free engine"


def _model_option(f: Any) -> Any:
    return click.option(
        "--model",
        "model_name",
        default=None,
        envvar="VECFORGE_MODEL",
        help=_MODEL_HELP,
    )(f)


def _parse_filters(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"--filters must be JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise click.BadParameter("--filters must be a JSON object")
    return parsed


@click.group()
@click.version_option(version=__version__, prog_name="VecForge")
def cli() -> None:
    """VecForge — Forge your vector database. Own it forever.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.
    """


# ─── Vault commands ───


@cli.command()
@click.argument("path")
@click.option("--vault", required=True, help="Path to vault database")
@click.option("--namespace", default="default", help="Target namespace")
@click.option("--chunk-size", default=1000, help="Chunk size in characters")
@click.option("--chunk-overlap", default=200, help="Chunk overlap in characters")
@_model_option
def ingest(
    path: str,
    vault: str,
    namespace: str,
    chunk_size: int,
    chunk_overlap: int,
    model_name: str | None,
) -> None:
    """Ingest documents from PATH into the vault.

    Supports: .txt, .md, .pdf, .docx, .html

    Example: vecforge ingest my_docs/ --vault my.db
    """
    from vecforge import VecForge

    click.echo(f"VecForge — Ingesting from {path}...")
    with VecForge(vault, model_name=model_name) as db:
        count = db.ingest(
            path,
            namespace=namespace,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    click.echo(f"✅ Ingested {count} chunks into vault '{vault}'")


@cli.command()
@click.argument("query")
@click.option("--vault", required=True, help="Path to vault database")
@click.option("--top-k", default=5, help="Number of results")
@click.option("--namespace", default=None, help="Restrict to namespace")
@click.option("--alpha", default=0.5, help="Semantic weight (0.0-1.0)")
@click.option("--rerank", is_flag=True, help="Enable cross-encoder reranking")
@click.option("--quantum", is_flag=True, help="Enable Grover-inspired reranking")
@click.option(
    "--filters",
    default=None,
    help='Metadata filters as JSON, e.g. \'{"year": {"gte": 2024}}\'',
)
@click.option("--recency", default=0.0, help="Recency weight (0.0-1.0)")
@click.option("--json", "as_json", is_flag=True, help="Print results as JSON")
@_model_option
def search(
    query: str,
    vault: str,
    top_k: int,
    namespace: str | None,
    alpha: float,
    rerank: bool,
    quantum: bool,
    filters: str | None,
    recency: float,
    as_json: bool,
    model_name: str | None,
) -> None:
    """Search the vault with a natural language query.

    Example: vecforge search "diabetes treatment" --vault my.db --quantum
    """
    from vecforge import VecForge

    with VecForge(vault, model_name=model_name) as db:
        results = db.search(
            query,
            top_k=top_k,
            namespace=namespace,
            alpha=alpha,
            rerank=rerank,
            quantum_rerank=quantum,
            filters=_parse_filters(filters),
            recency_weight=recency,
        )

    if as_json:
        click.echo(json.dumps([r.to_dict() for r in results], indent=2, default=str))
        return

    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        click.echo(f"\n{'─' * 60}")
        click.echo(f"Result {i} | Score: {r.score:.4f} | ID: {r.doc_id[:8]}...")
        click.echo(
            f"Namespace: {r.namespace} | Modality: {r.modality} | "
            f"Signals: {', '.join(r.signals) or '-'}"
        )
        if r.metadata:
            click.echo(f"Metadata: {json.dumps(r.metadata, default=str)}")
        click.echo(f"\n{r.text[:500]}")


@cli.command()
@click.argument("vault")
@_model_option
def stats(vault: str, model_name: str | None) -> None:
    """Show vault statistics.

    Example: vecforge stats my.db
    """
    from vecforge import VecForge

    with VecForge(vault, model_name=model_name) as db:
        info = db.stats()

    click.echo(f"\n{'═' * 50}")
    click.echo("VecForge Vault Statistics")
    click.echo(f"{'═' * 50}")
    click.echo(f"Path:           {info['path']}")
    click.echo(f"Version:        {info['version']}")
    click.echo(f"Documents:      {info['documents']}")
    click.echo(f"Encrypted:      {info['encrypted']}")
    click.echo(f"Quantum:        {info['quantum']}")
    click.echo(f"Protection:     {info['deletion_protection']}")
    click.echo(f"Namespaces:     {', '.join(info['namespaces'])}")
    click.echo(f"Index vectors:  {info['index_vectors']} ({info['index_type']})")
    click.echo(f"Tombstones:     {info['tombstones']}")
    click.echo(f"BM25 docs:      {info['bm25_documents']}")
    click.echo(f"Embedder:       {info['embedding_model']}")
    click.echo(f"\nBuilt by {info['built_by']}")


@cli.command()
@click.argument("vault")
@click.option("--format", "fmt", default="json", help="Export format (json)")
@click.option("--output", "-o", default=None, help="Output file path")
@click.option("--namespace", default=None, help="Export specific namespace")
def export(vault: str, fmt: str, output: str | None, namespace: str | None) -> None:
    """Export vault data to JSON.

    Example: vecforge export my.db -o data.json
    """
    from vecforge.core.storage import StorageBackend

    if fmt != "json":
        raise click.BadParameter("Only 'json' is supported", param_hint="--format")

    docs = []
    storage = StorageBackend(path=vault)
    for doc in storage.iter_docs(namespace=namespace):
        docs.append(
            {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "metadata": doc.metadata,
                "namespace": doc.namespace,
                "modality": doc.modality,
                "created_at": doc.created_at,
            }
        )
    storage.close()

    data = {"vault": vault, "documents": docs, "count": len(docs)}
    json_str = json.dumps(data, indent=2, default=str)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(json_str)
        click.echo(f"✅ Exported {len(docs)} documents to {output}")
    else:
        click.echo(json_str)


@cli.command()
@click.option("--vault", required=True, help="Path to vault database")
@click.option("--port", default=8080, help="Server port")
@click.option("--host", default="127.0.0.1", help="Server host (0.0.0.0 to expose)")
@click.option(
    "--api-key", default=None, envvar="VECFORGE_API_KEY", help="Require X-API-Key"
)
@_model_option
def serve(
    vault: str, port: int, host: str, api_key: str | None, model_name: str | None
) -> None:
    """Start VecForge REST API server.

    Example: vecforge serve --vault my.db --port 8080
    """
    click.echo(f"VecForge REST Server v{__version__} — {vault}")
    click.echo(
        f"Listening on {host}:{port}" + ("  (API key required)" if api_key else "")
    )
    click.echo("Built by Suneel Bose K · ArcGX TechLabs\n")

    import uvicorn

    from vecforge.server.app import create_app

    app = create_app(vault, api_key=api_key, model_name=model_name)
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.argument("vault")
@_model_option
def compact(vault: str, model_name: str | None) -> None:
    """Rebuild indexes without tombstones and refresh the snapshot.

    Example: vecforge compact my.db
    """
    from vecforge import VecForge

    with VecForge(vault, model_name=model_name) as db:
        before = db.stats()["tombstones"]
        db.compact()
        click.echo(
            f"✅ Compacted {vault}: removed {before} tombstones, {db.count()} docs"
        )


@cli.command()
@click.argument("vault")
@click.argument("destination")
def backup(vault: str, destination: str) -> None:
    """Write a consistent backup copy of the vault (safe while in use).

    Example: vecforge backup my.db backups/my-2026-10-17.db
    """
    from vecforge.core.storage import StorageBackend

    storage = StorageBackend(path=vault)
    try:
        storage.backup(destination)
    finally:
        storage.close()
    click.echo(f"✅ Backup written to {destination}")


# ─── MCP server ───


@cli.command()
@click.option(
    "--vault", default=None, help="Vault path (default ~/.vecforge/memory.db)"
)
@click.option("--agent", default="claude", help="Agent name for memory isolation")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http"]),
    default="stdio",
    help="stdio for Claude Desktop/Code, http for networked clients",
)
@click.option("--host", default="127.0.0.1", help="HTTP bind host")
@click.option("--port", default=8765, help="HTTP bind port")
@click.option(
    "--print-config",
    type=click.Choice(["claude-desktop", "claude-code"]),
    default=None,
    help="Print the client config snippet instead of starting the server",
)
@_model_option
def mcp(
    vault: str | None,
    agent: str,
    transport: str,
    host: str,
    port: int,
    print_config: str | None,
    model_name: str | None,
) -> None:
    """Start the local MCP memory server for Claude and other AI clients.

    Example: vecforge mcp --vault ~/.vecforge/memory.db --agent claude
    """
    if print_config:
        from vecforge.mcp.server import claude_code_config, claude_desktop_config

        cfg = (
            claude_desktop_config(vault, agent)
            if print_config == "claude-desktop"
            else claude_code_config(vault, agent)
        )
        click.echo(json.dumps(cfg, indent=2))
        return

    from vecforge.mcp.server import run_server

    run_server(
        vault_path=vault,
        agent=agent,
        transport=transport,
        host=host,
        port=port,
        model_name=model_name,
        encryption_key=os.environ.get("VECFORGE_KEY"),
    )


# ─── Memory commands ───


@cli.group()
def memory() -> None:
    """Manage AI memories (remember / recall / sessions)."""


def _memory_opts(f: Any) -> Any:
    f = click.option(
        "--vault", default=None, help="Vault path (default ~/.vecforge/memory.db)"
    )(f)
    f = click.option("--agent", default="claude", help="Agent name")(f)
    return _model_option(f)


@memory.command("remember")
@click.argument("content")
@click.option("--kind", default="fact", help="fact|preference|episode|task|note|…")
@click.option("--importance", default=0.5, help="0.0-1.0")
@click.option("--tags", default="", help="Comma-separated tags")
@_memory_opts
def memory_remember(
    content: str,
    kind: str,
    importance: float,
    tags: str,
    vault: str | None,
    agent: str,
    model_name: str | None,
) -> None:
    """Store a memory."""
    from vecforge.memory import MemoryStore

    with MemoryStore(vault, agent=agent, model_name=model_name) as mem:
        m = mem.remember(
            content,
            kind=kind,
            importance=importance,
            tags=[t.strip() for t in tags.split(",") if t.strip()],
        )
    verb = "Reinforced" if m.deduplicated else "Remembered"
    click.echo(f"✅ {verb} {m.memory_id} ({m.kind}, importance={m.importance:.2f})")


@memory.command("recall")
@click.argument("query")
@click.option("--top-k", default=5, help="Number of memories")
@click.option("--kinds", default="", help="Comma-separated kinds")
@click.option("--json", "as_json", is_flag=True, help="Print as JSON")
@_memory_opts
def memory_recall(
    query: str,
    top_k: int,
    kinds: str,
    as_json: bool,
    vault: str | None,
    agent: str,
    model_name: str | None,
) -> None:
    """Recall memories relevant to a query."""
    from vecforge.memory import MemoryStore

    kind_list = [k.strip() for k in kinds.split(",") if k.strip()] or None
    with MemoryStore(vault, agent=agent, model_name=model_name) as mem:
        memories = mem.recall(query, top_k=top_k, kinds=kind_list)
    if as_json:
        click.echo(json.dumps([m.to_dict() for m in memories], indent=2, default=str))
        return
    if not memories:
        click.echo("No memories found.")
        return
    for m in memories:
        click.echo(f"[{m.score:.3f}] ({m.kind}) {m.content}")


@memory.command("list")
@click.option("--limit", default=20, help="Maximum memories")
@click.option("--kinds", default="", help="Comma-separated kinds")
@_memory_opts
def memory_list(
    limit: int, kinds: str, vault: str | None, agent: str, model_name: str | None
) -> None:
    """List recent memories."""
    from vecforge.memory import MemoryStore

    kind_list = [k.strip() for k in kinds.split(",") if k.strip()] or None
    with MemoryStore(vault, agent=agent, model_name=model_name) as mem:
        memories = mem.list_memories(kinds=kind_list, limit=limit)
    if not memories:
        click.echo("No memories stored.")
        return
    for m in memories:
        click.echo(f"{m.memory_id}  ({m.kind}, {m.importance:.2f})  {m.content}")


@memory.command("forget")
@click.argument("memory_id")
@_memory_opts
def memory_forget(
    memory_id: str, vault: str | None, agent: str, model_name: str | None
) -> None:
    """Delete a memory by ID."""
    from vecforge.memory import MemoryStore

    with MemoryStore(vault, agent=agent, model_name=model_name) as mem:
        deleted = mem.forget(memory_id)
    click.echo("✅ Forgotten" if deleted else "Memory not found")
    if not deleted:
        sys.exit(1)


@memory.command("sessions")
@click.option("--limit", default=10, help="Maximum sessions")
@_memory_opts
def memory_sessions(
    limit: int, vault: str | None, agent: str, model_name: str | None
) -> None:
    """List recent sessions and summaries."""
    from vecforge.memory import MemoryStore

    with MemoryStore(vault, agent=agent, model_name=model_name) as mem:
        sessions = mem.list_sessions(limit=limit)
    if not sessions:
        click.echo("No sessions recorded.")
        return
    for s in sessions:
        state = "open" if s.is_open else "closed"
        click.echo(f"{s.session_id}  [{state}, {s.turn_count} memories]  {s.summary}")


@memory.command("context")
@click.option("--query", default="", help="Topic to focus on")
@click.option("--max-chars", default=4000, help="Character budget")
@_memory_opts
def memory_context(
    query: str, max_chars: int, vault: str | None, agent: str, model_name: str | None
) -> None:
    """Print a prompt-ready briefing from memory."""
    from vecforge.memory import MemoryStore

    with MemoryStore(vault, agent=agent, model_name=model_name) as mem:
        pack = mem.context_pack(query or None, max_chars=max_chars)
    click.echo(pack or "No memories stored yet.")


@memory.command("export")
@click.argument("output")
@_memory_opts
def memory_export(
    output: str, vault: str | None, agent: str, model_name: str | None
) -> None:
    """Export memories to a JSON file."""
    from vecforge.memory import MemoryStore

    with MemoryStore(vault, agent=agent, model_name=model_name) as mem:
        records = mem.export_json(output)
    click.echo(f"✅ Exported {len(records)} memories to {output}")


@memory.command("import")
@click.argument("source")
@_memory_opts
def memory_import(
    source: str, vault: str | None, agent: str, model_name: str | None
) -> None:
    """Import memories from a JSON export."""
    from vecforge.memory import MemoryStore

    with MemoryStore(vault, agent=agent, model_name=model_name) as mem:
        count = mem.import_json(source)
    click.echo(f"✅ Imported {count} memories from {source}")


if __name__ == "__main__":
    cli()

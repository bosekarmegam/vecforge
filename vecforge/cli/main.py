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

Provides commands for ingestion, search, statistics, and serving.

Built by Suneel Bose K · ArcGX TechLabs Private Limited.
"""

from __future__ import annotations

import json

import click


@click.group()
@click.version_option(version="1.0.0", prog_name="VecForge")
def cli() -> None:
    """VecForge — Forge your vector database. Own it forever.

    Built by Suneel Bose K · ArcGX TechLabs Private Limited.
    """


@cli.command()
@click.argument("path")
@click.option("--vault", required=True, help="Path to vault database")
@click.option("--namespace", default="default", help="Target namespace")
@click.option("--chunk-size", default=1000, help="Chunk size in characters")
@click.option("--chunk-overlap", default=200, help="Chunk overlap in characters")
def ingest(
    path: str,
    vault: str,
    namespace: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Ingest documents from PATH into the vault.

    Supports: .txt, .md, .pdf, .docx, .html

    Example: vecforge ingest my_docs/ --vault my.db
    """
    from vecforge import VecForge

    click.echo(f"VecForge — Ingesting from {path}...")
    with VecForge(vault) as db:
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
def search(
    query: str,
    vault: str,
    top_k: int,
    namespace: str | None,
    alpha: float,
    rerank: bool,
) -> None:
    """Search the vault with a natural language query.

    Example: vecforge search "diabetes treatment" --vault my.db
    """
    from vecforge import VecForge

    with VecForge(vault) as db:
        results = db.search(
            query,
            top_k=top_k,
            namespace=namespace,
            alpha=alpha,
            rerank=rerank,
        )

    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        click.echo(f"\n{'─' * 60}")
        click.echo(f"Result {i} | Score: {r.score:.4f} | ID: {r.doc_id[:8]}...")
        click.echo(f"Namespace: {r.namespace} | Modality: {r.modality}")
        if r.metadata:
            click.echo(f"Metadata: {json.dumps(r.metadata, default=str)}")
        click.echo(f"\n{r.text[:500]}")


@cli.command()
@click.argument("vault")
def stats(vault: str) -> None:
    """Show vault statistics.

    Example: vecforge stats my.db
    """
    from vecforge import VecForge

    with VecForge(vault) as db:
        info = db.stats()

    click.echo(f"\n{'═' * 50}")
    click.echo("VecForge Vault Statistics")
    click.echo(f"{'═' * 50}")
    click.echo(f"Path:           {info['path']}")
    click.echo(f"Documents:      {info['documents']}")
    click.echo(f"Encrypted:      {info['encrypted']}")
    click.echo(f"Quantum:        {info['quantum']}")
    click.echo(f"Protection:     {info['deletion_protection']}")
    click.echo(f"Namespaces:     {', '.join(info['namespaces'])}")
    click.echo(f"Index vectors:  {info['index_vectors']}")
    click.echo(f"BM25 docs:      {info['bm25_documents']}")
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

    docs = []
    storage = StorageBackend(path=vault)
    all_docs = storage.get_all_docs(namespace=namespace)

    for doc in all_docs:
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
@click.option("--host", default="0.0.0.0", help="Server host")
def serve(vault: str, port: int, host: str) -> None:
    """Start VecForge REST API server.

    Example: vecforge serve --vault my.db --port 8080
    """
    click.echo(f"VecForge REST Server — {vault}")
    click.echo(f"Listening on {host}:{port}")
    click.echo("Built by Suneel Bose K · ArcGX TechLabs\n")

    import uvicorn

    from vecforge.server.app import create_app

    app = create_app(vault)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()

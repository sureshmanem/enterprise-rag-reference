"""Command-line interface: `rag ingest <path>`, `rag ask "<question>"`, `rag serve`."""

from __future__ import annotations

import json
import logging

import typer

from .config import get_settings
from .loaders import SharePointLoader, load_directory
from .pipeline import RAG

app = typer.Typer(add_completion=False, help="rag-mini: Retrieval-Augmented Generation demo")


@app.callback()
def _setup(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")


@app.command()
def ingest(path: str = typer.Argument(..., help="File or folder with PDF/DOCX/MD/TXT")) -> None:
    """Parse, chunk, embed and index local documents."""
    stats = RAG(get_settings()).ingest(load_directory(path))
    typer.echo(json.dumps(stats, indent=2))


@app.command("ingest-sharepoint")
def ingest_sharepoint(
    hostname: str = typer.Option(..., help="e.g. contoso.sharepoint.com"),
    site: str = typer.Option(..., help="e.g. sites/HR"),
    folder: str = typer.Option("", help="Folder inside the default document library"),
) -> None:
    """Index a SharePoint document library (needs SP_TENANT_ID/SP_CLIENT_ID/SP_CLIENT_SECRET)."""
    stats = RAG(get_settings()).ingest(SharePointLoader(hostname, site, folder).load())
    typer.echo(json.dumps(stats, indent=2))


@app.command()
def ask(question: str, as_json: bool = typer.Option(False, "--json")) -> None:
    """Ask a question against the indexed documents."""
    ans = RAG(get_settings()).ask(question)
    if as_json:
        typer.echo(ans.model_dump_json(indent=2))
        return
    typer.echo(ans.answer)
    for c in ans.citations:
        page = f", page {c.page}" if c.page else ""
        typer.echo(f"  [{c.ref}] {c.source}{page}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the HTTP API with a minimal chat UI."""
    import uvicorn

    uvicorn.run("ragmini.api:app", host=host, port=port)


if __name__ == "__main__":
    app()

"""
IPChat CLI - Command-line interface for Bronchmonkey.
"""

import os
import sys
import click
import logging
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

from ipchat.core.config import load_config, Edition

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, debug):
    """
    IPChat - Bronchmonkey Interventional Pulmonology Research Assistant
    
    Run different editions with: ipchat run --edition [full|lite|space]
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


@cli.command()
@click.option(
    "--edition",
    type=click.Choice(["full", "lite", "space"], case_sensitive=False),
    default="lite",
    help="Edition to run (full/lite/space)"
)
@click.option(
    "--ui",
    type=click.Choice(["streamlit", "api", "both"], case_sensitive=False),
    default="streamlit",
    help="UI mode to run"
)
@click.option("--host", default=None, help="Override host address")
@click.option("--port", type=int, default=None, help="Override port")
@click.option("--config", type=click.Path(exists=True), help="Config file path")
@click.pass_context
def run(ctx, edition, ui, host, port, config):
    """Run IPChat in the specified edition."""
    
    console.print(f"[bold green]🐵 Starting Bronchmonkey ({edition} edition)...[/]")
    
    # Load configuration
    config_path = Path(config) if config else None
    cfg = load_config(edition=edition, config_file=config_path)
    
    # Override host/port if provided
    if host:
        cfg.infra.ui_host = host
        cfg.infra.api_host = host
    if port:
        if ui == "api":
            cfg.infra.api_port = port
        else:
            cfg.infra.ui_port = port
    
    # Run based on UI mode
    if ui == "streamlit":
        run_streamlit(cfg)
    elif ui == "api":
        run_api(cfg)
    elif ui == "both":
        run_full_stack(cfg)


def run_streamlit(config):
    """Run Streamlit UI."""
    import subprocess
    
    # Set environment variables
    os.environ["IPCHAT_EDITION"] = config.edition.value
    
    # Determine which app to run
    app_map = {
        Edition.FULL: "ipchat/apps/streamlit_full.py",
        Edition.LITE: "ipchat/apps/streamlit_lite.py",
        Edition.SPACE: "ipchat/apps/space_app.py",
    }
    
    app_file = app_map.get(config.edition, "ipchat/apps/streamlit_lite.py")
    
    console.print(f"[cyan]Starting Streamlit on {config.infra.ui_host}:{config.infra.ui_port}[/]")
    
    cmd = [
        "streamlit", "run",
        app_file,
        "--server.address", config.infra.ui_host,
        "--server.port", str(config.infra.ui_port),
        "--theme.base", "dark"
    ]
    
    subprocess.run(cmd)


def run_api(config):
    """Run FastAPI server."""
    import uvicorn
    
    console.print(f"[cyan]Starting API on {config.infra.api_host}:{config.infra.api_port}[/]")
    
    uvicorn.run(
        "ipchat.api.server:app",
        host=config.infra.api_host,
        port=config.infra.api_port,
        reload=config.debug_mode
    )


def run_full_stack(config):
    """Run both API and UI (full edition only)."""
    if config.edition != Edition.FULL:
        console.print("[red]Full stack mode is only available in 'full' edition[/]")
        sys.exit(1)
    
    import subprocess
    import threading
    
    # Start API in background thread
    api_thread = threading.Thread(target=run_api, args=(config,))
    api_thread.daemon = True
    api_thread.start()
    
    # Run Streamlit in main thread
    run_streamlit(config)


@cli.command()
@click.option(
    "--source",
    type=click.Path(exists=True),
    required=True,
    help="Source directory with documents"
)
@click.option(
    "--output",
    type=click.Path(),
    default="./data/index",
    help="Output directory for indexes"
)
@click.option("--rebuild", is_flag=True, help="Rebuild from scratch")
@click.pass_context
def index(ctx, source, output, rebuild):
    """Build or update search indexes."""
    
    console.print("[bold yellow]🔨 Building indexes...[/]")
    
    from ipchat.core.indexing import IndexBuilder
    
    builder = IndexBuilder(
        source_dir=Path(source),
        output_dir=Path(output),
        rebuild=rebuild
    )
    
    with console.status("[cyan]Processing documents..."):
        stats = builder.build()
    
    # Display results
    table = Table(title="Indexing Complete")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in stats.items():
        table.add_row(key, str(value))
    
    console.print(table)


@cli.command()
@click.pass_context
def verify(ctx):
    """Verify installation and configuration."""
    
    console.print("[bold]🔍 Verifying IPChat installation...[/]")
    
    checks = []
    
    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    checks.append(("Python Version", py_version, py_version >= "3.10"))
    
    # Check required packages
    try:
        import streamlit
        checks.append(("Streamlit", "✓ Installed", True))
    except ImportError:
        checks.append(("Streamlit", "✗ Not installed", False))
    
    try:
        import faiss
        checks.append(("FAISS", "✓ Installed", True))
    except ImportError:
        checks.append(("FAISS", "✗ Not installed", False))
    
    # Check data files
    data_dir = Path("./data")
    if data_dir.exists():
        index_files = list(data_dir.glob("index/*.index"))
        checks.append(("Index Files", f"{len(index_files)} found", len(index_files) > 0))
    else:
        checks.append(("Index Files", "Data directory not found", False))
    
    # Check environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    checks.append(("OpenAI API Key", "✓ Set" if openai_key else "✗ Not set", bool(openai_key)))
    
    # Display results
    table = Table(title="System Verification")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    all_good = True
    for name, status, ok in checks:
        style = "green" if ok else "red"
        table.add_row(name, f"[{style}]{status}[/]")
        all_good = all_good and ok
    
    console.print(table)
    
    if all_good:
        console.print("\n[bold green]✅ All checks passed![/]")
    else:
        console.print("\n[bold red]⚠️ Some checks failed. Please review the issues above.[/]")
        sys.exit(1)


@cli.command()
@click.option(
    "--format",
    type=click.Choice(["json", "csv", "parquet"], case_sensitive=False),
    default="json",
    help="Export format"
)
@click.option(
    "--output",
    type=click.Path(),
    required=True,
    help="Output file path"
)
@click.pass_context
def export(ctx, format, output):
    """Export indexed data to various formats."""
    
    console.print(f"[cyan]📤 Exporting data to {format} format...[/]")
    
    from ipchat.core.export import DataExporter
    
    exporter = DataExporter()
    exporter.export(format=format, output_path=Path(output))
    
    console.print(f"[green]✓ Data exported to {output}[/]")


@cli.command()
def info():
    """Display information about the current configuration."""
    
    config = load_config()
    
    console.print("[bold]Bronchmonkey Configuration[/]\n")
    
    table = Table(show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")
    
    table.add_row("Edition", config.edition.value)
    table.add_row("LLM Provider", config.llm.provider)
    table.add_row("LLM Model", config.llm.model)
    table.add_row("Vector Store", config.retrieval.vector_store)
    table.add_row("Keyword Index", config.retrieval.keyword_index)
    table.add_row("SQL Backend", config.infra.sql_backend or "None")
    table.add_row("API Port", str(config.infra.api_port))
    table.add_row("UI Port", str(config.infra.ui_port))
    table.add_row("Auth Enabled", str(config.auth.enabled))
    table.add_row("Depth Features", str(config.depth_features))
    
    console.print(table)


def main():
    """Main entry point."""
    try:
        cli(obj={})
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        if "--debug" in sys.argv:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
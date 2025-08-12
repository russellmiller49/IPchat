import typer
from pathlib import Path
from ipchat.extract.textbook.pipeline import extract_textbook

app = typer.Typer(no_args_is_help=True)

@app.command("extract-textbook")
def extract_textbook_cmd(
    pdf: Path,
    adobe_json: Path,
    title: str = "",
    out: Path = Path("outputs")
):
    out.mkdir(parents=True, exist_ok=True)
    result = extract_textbook(pdf, adobe_json, title or pdf.stem)
    out_file = out / f"{pdf.stem}.textbook.json"
    out_file.write_text(result.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"Saved: {out_file}")

if __name__ == "__main__":
    app()
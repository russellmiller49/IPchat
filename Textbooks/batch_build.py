
import json
from pathlib import Path
try:
    import yaml
except Exception:
    yaml = None

from build_chapter import assemble_chapter

def main(cfg_path: Path):
    if yaml is None:
        raise SystemExit("Please 'pip install pyyaml' to use batch_build.py")
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    book = cfg["book"]
    for ch in cfg["chapters"]:
        pdf = Path(ch["pdf"])
        out = pdf.with_suffix(".chapter.json")
        doc = assemble_chapter(
            pdf_path=pdf,
            chapter_title=ch["title"],
            authors=ch["authors"],
            book_title=book.get("title",""),
            publisher=book.get("publisher",""),
            isbn13=book.get("isbn13"),
            chapter_number=str(ch.get("number","")),
            source_url=ch.get("doi")
        )
        out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Wrote", out)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    args = ap.parse_args()
    main(args.config)

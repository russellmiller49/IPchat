import json
from pathlib import Path
from typing import Dict, Any, List

def load_adobe_extract(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def parse_tables(adobe: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for t in adobe.get("tables", []):
        out.append({
            "table_id": str(t.get("ObjectID") or ""),
            "page": t.get("Page"),
            "bbox": (t.get("attributes") or {}).get("BBox"),
            "source_xlsx": (t.get("filePaths") or [None])[0],
        })
    return out

def parse_figures(adobe: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for f in adobe.get("figures", []):
        out.append({
            "figure_id": str(f.get("ObjectID") or ""),
            "page": f.get("Page"),
            "bbox": (f.get("attributes") or {}).get("BBox"),
            "asset_path": (f.get("filePaths") or [None])[0],
        })
    return out

def flatten_text_units(adobe: Dict[str, Any]) -> str:
    parts = []
    for unit in adobe.get("content", {}).get("text_units", []):
        p = unit.get("provenance", {}).get("page")
        txt = unit.get("text", "")
        if p:
            parts.append(f"\n[PAGE {p}]\n{txt}")
        else:
            parts.append(txt)
    return "".join(parts)
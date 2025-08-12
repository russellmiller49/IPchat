#!/usr/bin/env python3
import json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
ARCHIVE = ROOT / "archive"
CANON_KEEP = {"medical_evidence_oe_final.schema.json"}  # will be moved/renamed already

def plan():
    moves = []
    if SCHEMAS.exists():
        for p in SCHEMAS.glob("*.schema.json"):
            if p.name not in CANON_KEEP:
                moves.append((p, ARCHIVE / "schemas" / p.name))
    return moves

def main(dry_run=True):
    todo = plan()
    report = []
    for src, dst in todo:
        dst.parent.mkdir(parents=True, exist_ok=True)
        report.append({"src": str(src), "dst": str(dst)})
        if not dry_run:
            shutil.move(str(src), str(dst))
    print(json.dumps({"dry_run": dry_run, "moves": report}, indent=2))

if __name__ == "__main__":
    main(dry_run=True)
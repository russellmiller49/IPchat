#!/usr/bin/env python3
"""
Load trials and chapters JSON into Postgres.

Usage:
  export DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/db
  python ingestion/load_json_to_pg.py \
      --trials-dir data/complete_extractions \
      --chapters-dir data/Textbooks

Notes:
  - This script is schema-tolerant: it looks for common keys and falls back gracefully.
  - Run sql/schema.sql before loading.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@dataclass
class StudyRow:
    study_id: str
    title: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    nct_id: Optional[str] = None
    rob_overall: Optional[str] = None


def yield_trials(trials_dir: Path) -> Iterable[Dict[str, Any]]:
    for p in sorted(trials_dir.glob("**/*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            yield p, data
        except Exception as e:
            print(f"[WARN] Skip {p}: {e}")


def to_study_row(doc_id: str, data: Dict[str, Any]) -> StudyRow:
    # Handle both root-level metadata and document.metadata structure
    meta = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    if not meta and "document" in data and isinstance(data["document"], dict):
        meta = data["document"].get("metadata", {})
    
    return StudyRow(
        study_id=meta.get("doi") or meta.get("nct_id") or doc_id,
        title=meta.get("title"),
        year=(int(meta.get("year")) if str(meta.get("year")).isdigit() else None),
        journal=meta.get("journal"),
        doi=meta.get("doi"),
        nct_id=meta.get("nct_id"),
        rob_overall=(data.get("risk_of_bias", {}) or {}).get("overall")
    )


def insert_study(engine: Engine, row: StudyRow) -> str:
    sql = text(
        """
        INSERT INTO studies (study_id, title, year, journal, doi, nct_id, rob_overall)
        VALUES (:study_id, :title, :year, :journal, :doi, :nct_id, :rob_overall)
        ON CONFLICT (study_id) DO UPDATE SET
          title = EXCLUDED.title,
          year = EXCLUDED.year,
          journal = EXCLUDED.journal,
          doi = EXCLUDED.doi,
          nct_id = EXCLUDED.nct_id,
          rob_overall = EXCLUDED.rob_overall
        RETURNING study_id
        """
    )
    with engine.begin() as conn:
        sid = conn.execute(sql, vars(row)).scalar()
    return sid


def insert_arm(engine: Engine, study_id: str, arm_id: str, arm: Dict[str, Any]):
    sql = text(
        """
        INSERT INTO arms (study_id, arm_id, name, n_randomized, n_analyzed, n_completed)
        VALUES (:study_id, :arm_id, :name, :n_randomized, :n_analyzed, :n_completed)
        ON CONFLICT (study_id, arm_id) DO UPDATE SET
          name = EXCLUDED.name,
          n_randomized = EXCLUDED.n_randomized,
          n_analyzed = EXCLUDED.n_analyzed,
          n_completed = EXCLUDED.n_completed
        """
    )
    payload = {
        "study_id": study_id,
        "arm_id": arm_id,
        "name": arm.get("name") or arm_id,
        "n_randomized": arm.get("n_randomized"),
        "n_analyzed": arm.get("n_analyzed"),
        "n_completed": arm.get("n_completed"),
    }
    with engine.begin() as conn:
        conn.execute(sql, payload)


def insert_outcome(engine: Engine, study_id: str, row: Dict[str, Any]):
    sql = text(
        """
        INSERT INTO outcomes (
          study_id, concept_id, name, outcome_type, timepoint_iso8601, ref_arm_id,
          measure, est, ci_lower, ci_upper, ci_level, p_value, p_operator,
          adjusted, unit, events_treat, total_treat, events_ctrl, total_ctrl,
          pages, table_ref
        ) VALUES (
          :study_id, :concept_id, :name, :outcome_type, :timepoint_iso8601, :ref_arm_id,
          :measure, :est, :ci_lower, :ci_upper, :ci_level, :p_value, :p_operator,
          :adjusted, :unit, :events_treat, :total_treat, :events_ctrl, :total_ctrl,
          :pages, :table_ref
        )
        ON CONFLICT (study_id, concept_id, measure, timepoint_iso8601) DO UPDATE SET
          est = EXCLUDED.est,
          ci_lower = EXCLUDED.ci_lower,
          ci_upper = EXCLUDED.ci_upper,
          p_value = EXCLUDED.p_value,
          unit = EXCLUDED.unit,
          pages = EXCLUDED.pages,
          table_ref = EXCLUDED.table_ref
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, row)


def insert_safety(engine: Engine, study_id: str, row: Dict[str, Any]):
    sql = text(
        """
        INSERT INTO safety (
          study_id, soc, pt, serious, period, arm_id, patients, events, percentage, pages
        ) VALUES (
          :study_id, :soc, :pt, :serious, :period, :arm_id, :patients, :events, :percentage, :pages
        )
        ON CONFLICT (study_id, pt, period, arm_id) DO UPDATE SET
          patients = EXCLUDED.patients,
          events = EXCLUDED.events,
          percentage = EXCLUDED.percentage,
          pages = EXCLUDED.pages
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, row)


def load_trials(engine: Engine, trials_dir: Path):
    for path, data in yield_trials(trials_dir):
        # Choose an id: doi|nct|file-stem
        doc_id = path.stem
        srow = to_study_row(doc_id, data)
        study_id = insert_study(engine, srow)

        # Arms (if present)
        arms = data.get("population", {}).get("arms") or data.get("arms") or []
        if isinstance(arms, dict):
            arms = [{"arm_id": k, **(v or {})} for k, v in arms.items()]
        for arm in arms:
            arm_id = str(arm.get("arm_id") or arm.get("id") or arm.get("name") or "A")
            insert_arm(engine, study_id, arm_id, arm)

        # Outcomes (if present)
        outcomes = (data.get("outcomes") or {}).copy()
        for group in ["primary", "secondary", "other"]:
            for o in outcomes.get(group, []) or []:
                row = {
                    "study_id": study_id,
                    "concept_id": str(o.get("concept_id") or o.get("name") or "unk"),
                    "name": o.get("name"),
                    "outcome_type": o.get("type") or o.get("outcome_type"),
                    "timepoint_iso8601": o.get("timepoint_iso8601") or o.get("timepoint"),
                    "ref_arm_id": o.get("ref_arm_id") or o.get("comparison") or None,
                    "measure": o.get("measure"),
                    "est": o.get("difference") or o.get("est"),
                    "ci_lower": (o.get("ci", {}) or {}).get("lower") or o.get("ci_lower"),
                    "ci_upper": (o.get("ci", {}) or {}).get("upper") or o.get("ci_upper"),
                    "ci_level": (o.get("ci", {}) or {}).get("level") or o.get("ci_level"),
                    "p_value": o.get("p_value"),
                    "p_operator": o.get("p_operator"),
                    "adjusted": bool(o.get("adjusted")) if o.get("adjusted") is not None else None,
                    "unit": o.get("unit"),
                    "events_treat": o.get("events_treat"),
                    "total_treat": o.get("total_treat"),
                    "events_ctrl": o.get("events_ctrl"),
                    "total_ctrl": o.get("total_ctrl"),
                    "pages": o.get("pages") or [],
                    "table_ref": o.get("table") or o.get("table_ref"),
                }
                insert_outcome(engine, study_id, row)

        # Safety (if present)
        for ae in data.get("adverse_events") or []:
            row = {
                "study_id": study_id,
                "soc": ae.get("soc"),
                "pt": ae.get("event") or ae.get("pt"),
                "serious": bool(ae.get("serious")) if ae.get("serious") is not None else None,
                "period": ae.get("period"),
                "arm_id": ae.get("arm_id"),
                "patients": ae.get("patients"),
                "events": ae.get("events"),
                "percentage": ae.get("intervention_percent") or ae.get("percentage"),
                "pages": ae.get("pages") or [],
            }
            insert_safety(engine, study_id, row)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-dir", type=Path, required=False, default=Path("data/complete_extractions"))
    parser.add_argument("--chapters-dir", type=Path, required=False, help="Optional chapters JSON dir")
    args = parser.parse_args()

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise SystemExit("Set DATABASE_URL env var (e.g., postgresql+psycopg2://user:pass@host/db)")
    engine = create_engine(db_url, future=True)

    print(f"Loading trials from {args.trials_dir}…")
    load_trials(engine, args.trials_dir)
    print("Done.")

    # Chapters can be indexed as chunks directly (see chunker.py). Optional relational loading omitted here.

if __name__ == "__main__":
    main()


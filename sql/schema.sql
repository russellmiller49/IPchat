-- Postgres schema for studies, arms, outcomes, and safety
-- Run with: psql "$DATABASE_URL" -f sql/schema.sql

CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS studies (
  study_id TEXT PRIMARY KEY,
  title TEXT,
  year INT,
  journal TEXT,
  doi TEXT,
  nct_id TEXT,
  rob_overall TEXT
);

CREATE TABLE IF NOT EXISTS arms (
  study_id TEXT REFERENCES studies(study_id) ON DELETE CASCADE,
  arm_id TEXT,
  name TEXT,
  n_randomized INT,
  n_analyzed INT,
  n_completed INT,
  PRIMARY KEY (study_id, arm_id)
);

-- One row per outcome measure/timepoint/comparison
CREATE TABLE IF NOT EXISTS outcomes (
  study_id TEXT REFERENCES studies(study_id) ON DELETE CASCADE,
  concept_id TEXT,
  name TEXT,
  outcome_type TEXT,              -- binary | continuous
  timepoint_iso8601 TEXT,         -- e.g., P12M
  ref_arm_id TEXT,
  measure TEXT,                   -- risk_difference | mean_diff | rr | or | smd
  est DOUBLE PRECISION,
  ci_lower DOUBLE PRECISION,
  ci_upper DOUBLE PRECISION,
  ci_level DOUBLE PRECISION,
  p_value DOUBLE PRECISION,
  p_operator TEXT,                -- '<' or '='
  adjusted BOOLEAN,
  unit TEXT,
  events_treat INT,
  total_treat INT,
  events_ctrl INT,
  total_ctrl INT,
  pages INT[],
  table_ref TEXT,
  PRIMARY KEY (study_id, concept_id, measure, timepoint_iso8601)
);

CREATE TABLE IF NOT EXISTS safety (
  study_id TEXT REFERENCES studies(study_id) ON DELETE CASCADE,
  soc TEXT,
  pt TEXT,
  serious BOOLEAN,
  period TEXT,                    -- e.g., 0–45d, 46–365d
  arm_id TEXT,
  patients INT,
  events INT,
  percentage DOUBLE PRECISION,
  pages INT[],
  PRIMARY KEY (study_id, pt, period, arm_id)
);

-- Optional: chunks registry for BM25/metadata mirroring
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  document_id TEXT,
  source TEXT,                    -- trial | chapter
  pages INT[],
  section_path TEXT[],
  table_number TEXT,
  figure_number TEXT,
  trial_signals JSONB,
  text TEXT
);

-- Useful indexes
CREATE INDEX IF NOT EXISTS idx_outcomes_intervention ON outcomes (ref_arm_id);
CREATE INDEX IF NOT EXISTS idx_outcomes_timepoint ON outcomes (timepoint_iso8601);
CREATE INDEX IF NOT EXISTS idx_outcomes_name_gin ON outcomes USING gin (name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_studies_year ON studies (year);
CREATE INDEX IF NOT EXISTS idx_safety_pt ON safety (pt);

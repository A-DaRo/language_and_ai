# Data Ingestion Steps: SOBR Assignment Data

| Metadata | Value |
| :--- | :--- |
| **Document Type** | Planning |
| **Version** | 1.0 |
| **Last Updated** | 2025-02-14 |
| **Project** | Neuro-Symbolic Stylometry Pipeline |
| **Related Documents** | `Misc/Interim_Assignment/Generated/phaseA-D_implementation_plan.md`, `Misc/Interim_Assignment/SOBR.md` |

---

## Index

- Index (this list)
- Executive Summary
- 1. Data Inventory and Observations
- 2. Target Unified Schema
- 3. Ingestion Workflow
- 4. Validation and Quality Assurance
- 5. Artifacts and Handover
- 6. Risks and Mitigations
- 7. Alignment with Implementation Plan

## Executive Summary

This document specifies an end-to-end plan to ingest the SOBR assignment data from eight CSV files into a single, memory-mapped Apache Arrow dataset. The plan addresses schema consolidation, normalization of a misspelled identifier column, and careful handling of long, user-generated text. It defines validation steps to preserve label integrity and to ensure author-stratified splits without leakage. The output will be a unified Arrow file and split metadata, aligned with Phase A requirements in the implementation plan.

## 1. Data Inventory and Observations

**Data location:** `assignment_data/`

**Files and row counts (excluding headers):**

| File | Rows | Columns (as observed) | Label Column |
| :--- | ---: | :--- | :--- |
| `assignment_data/birth_year.csv` | 41,873 | `auhtor_ID`, `post`, `birth_year` | `birth_year` |
| `assignment_data/extrovert_introvert.csv` | 40,452 | `auhtor_ID`, `post`, `extrovert` | `extrovert` |
| `assignment_data/feeling_thinking.csv` | 39,600 | `auhtor_ID`, `post`, `feeling` | `feeling` |
| `assignment_data/gender.csv` | 44,635 | `auhtor_ID`, `post`, `female` | `female` |
| `assignment_data/judging_perceiving.csv` | 41,365 | `auhtor_ID`, `post`, `judging` | `judging` |
| `assignment_data/nationality.csv` | 82,616 | `auhtor_ID`, `post`, `nationality` | `nationality` |
| `assignment_data/political_leaning.csv` | 57,231 | `auhtor_ID`, `post`, `political_leaning` | `political_leaning` |
| `assignment_data/sensing_intuitive.csv` | 41,839 | `auhtor_ID`, `post`, `sensing` | `sensing` |

**Key observations:**

- The identifier column is misspelled as `auhtor_ID` in all files and must be normalized to `author_id`.
- Each file contains a `post` field with long, quoted text and punctuation; robust CSV parsing is required to preserve content without truncation.
- Label columns are single attributes per file; the unified schema must allow nullable columns for all labels.
- Non-ASCII characters are present in the text; ingestion must preserve UTF-8 encoding.

## 2. Target Unified Schema

**Unified dataset fields (minimum set):**

| Field | Type | Source | Notes |
| :--- | :--- | :--- | :--- |
| `author_id` | string (dictionary-encoded) | `auhtor_ID` | Normalize misspelling across files |
| `post` | string | `post` | Preserve exact text |
| `birth_year` | int32 (nullable) | `birth_year.csv` | Numeric year; null elsewhere |
| `female` | int8/bool (nullable) | `gender.csv` | Binary label; null elsewhere |
| `extrovert` | int8/bool (nullable) | `extrovert_introvert.csv` | Binary label; null elsewhere |
| `feeling` | int8/bool (nullable) | `feeling_thinking.csv` | Binary label; null elsewhere |
| `judging` | int8/bool (nullable) | `judging_perceiving.csv` | Binary label; null elsewhere |
| `sensing` | int8/bool (nullable) | `sensing_intuitive.csv` | Binary label; null elsewhere |
| `nationality` | string (dictionary-encoded, nullable) | `nationality.csv` | Categorical |
| `political_leaning` | string (dictionary-encoded, nullable) | `political_leaning.csv` | Categorical |

**Notes:**

- The ingestion plan aligns with Phase A requirements for dictionary encoding and memory-mapped Arrow storage.
- If additional metadata (e.g., `post_id`, `subreddit`, `created_on`) is required, it must be sourced externally; it is not present in the assignment data.

## 3. Ingestion Workflow

1. **File discovery and integrity checks**
   - Confirm all eight CSV files exist in `assignment_data/`.
   - Validate that headers match the observed pattern and that row counts match expectations.
2. **Schema normalization**
   - Rename `auhtor_ID` to `author_id` in each file.
   - Cast label columns to the target types (integers or strings as appropriate).
3. **Merge and consolidation**
   - Perform a full outer merge on `(author_id, post)` across all eight files.
   - Ensure that missing labels remain null rather than defaulting to zeros.
4. **Arrow conversion and storage**
   - Convert the consolidated table to Apache Arrow with dictionary encoding for categorical columns.
   - Write as Arrow IPC (Feather v2) using LZ4 compression to `artifacts/data/sobr_unified.arrow`.
5. **Memory-mapped loading setup**
   - Validate memory-mapped loading with `memory_map=True` or `keep_in_memory=False`.
6. **Author-stratified split preparation**
   - Generate train/validation/test splits without author overlap.
   - Store split metadata (indices, counts, and random seed) in a companion JSON file.

## 4. Validation and Quality Assurance

- **Schema validation:** Verify column names and types match the unified schema.
- **Row count reconciliation:** Sum and reconcile counts per label source and ensure no unexpected row inflation after merge.
- **Null pattern checks:** Confirm that labels are present only where sourced and null elsewhere.
- **Label value checks:** Validate label domains (binary labels in {0,1}; categorical labels in known sets).
- **Text integrity checks:** Randomly sample posts to confirm that punctuation and quoted text are preserved.
- **Author leakage checks:** Confirm no author_id overlap between train/validation/test splits.

## 5. Artifacts and Handover

**Primary artifacts:**

- `artifacts/data/sobr_unified.arrow` (unified dataset)
- `artifacts/data/splits.json` (author-stratified split metadata)
- `artifacts/data/ingestion_report.md` (row counts, schema summary, label distributions)

**Handover contract to Phase A:**

- The dataset must include the `post` column for masking, and all label columns for downstream evaluation.

## 6. Risks and Mitigations

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| Misspelled identifier (`auhtor_ID`) | Join failures or duplicate authors | Normalize to `author_id` during ingestion |
| CSV quoting and long text | Truncated or corrupted `post` values | Use a robust CSV parser with full quoting support |
| Non-ASCII characters | Lossy encoding | Enforce UTF-8 reads and writes |
| Duplicate `(author_id, post)` rows | Label conflicts | Deduplicate and log conflicts before merge |
| Memory pressure on merge | Slow or failed conversion | Use chunked merging and streaming writes |

## 7. Alignment with Implementation Plan

- **FR-01 to FR-05 (Data Ingestion Requirements):** The plan directly implements Pandas-to-Arrow conversion, schema consolidation, memory-mapped I/O, dictionary encoding, and author ID indexing.
- **PA-T01 to PA-T05 (Phase A tasks):** The schema definition, converter, dataset wrapper, stratified splitting, and DataLoader readiness are explicitly covered.


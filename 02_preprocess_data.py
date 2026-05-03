"""
02 — beat the raw exports into one shape you can embed.

We clean text (PII-ish noise → placeholders), pull error codes, score row quality,
merge sources, and write the CSV/JSON our later scripts expect.

Typical runs:
  python 02_preprocess_data.py
  python 02_preprocess_data.py --new-source ./data/raw/servicenow --dataset-name servicenow --append
"""
from __future__ import annotations

import re
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

from department_mapping import DEPARTMENTS, map_to_department

RAW_DIR = Path("./data/raw")
PROCESSED_DIR = Path("./data/processed")
KB_DIR = Path("./data/knowledge_base")
_MAJOR_CANDIDATES = (
    PROCESSED_DIR / "its_ticket80.csv",
    PROCESSED_DIR / "its_tickets_80k.csv",
)


def default_major_its_csv_path() -> Path:
    for p in _MAJOR_CANDIDATES:
        if p.exists():
            return p
    return _MAJOR_CANDIDATES[-1]


MAJOR_TICKETS_PATH = default_major_its_csv_path()
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --- text cleanup for embedding ---
def clean_text(text):
    """messy tickets need love: urls/emails/paths → tokens so embeddings don't chase noise"""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'http[s]?://\S+', '[URL]', text)
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
    text = re.sub(r'[A-Za-z]:\\[\w\\\.]+', '[PATH]', text)
    text = re.sub(r'/[\w/\.]+', '[PATH]', text)
    text = re.sub(r'0x[0-9a-fA-F]+', '[HEX]', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()


def extract_error_codes(text):
    """grabs stuff like 0x..., TICKET-123, errno — BM25 likes these"""
    if not isinstance(text, str):
        return []
    patterns = [
        r'(?:error|code|status)\s*[:\-]?\s*(\d{3,5})',       # error 404, code: 500
        r'(0x[0-9a-fA-F]{4,8})',                              # hex error codes
        r'([A-Z]{2,5}[\-_]\d{3,6})',                          # TICKET-12345 style
        r'(?:errno|E)\s*(\d{1,4})',                            # errno 13
    ]
    codes = []
    for pattern in patterns:
        codes.extend(re.findall(pattern, text, re.IGNORECASE))
    return list(set(codes))


def compute_text_quality_score(row):
    """janky but fast 0–1 score — we use it to filter junk rows when appending"""
    score = 0.0
    if row.get("title") and len(str(row["title"])) > 5:
        score += 0.3
    if row.get("description") and len(str(row["description"])) > 20:
        score += 0.4
    if row.get("component") and str(row["component"]).strip():
        score += 0.15
    if row.get("severity") and str(row["severity"]).strip():
        score += 0.075
    if row.get("resolution") and len(str(row["resolution"])) > 5:
        score += 0.075
    return round(min(score, 1.0), 3)


def process_synthetic_tickets():
    path = PROCESSED_DIR / "synthetic_tickets.csv"
    if not path.exists():
        print("  ⚠ synthetic_tickets.csv not found. Run 01_download_data.py first.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"  Loaded synthetic: {len(df)} tickets")

    df["title_clean"] = df["title"].apply(clean_text)
    df["description_clean"] = df["description"].apply(clean_text)
    df["resolution_clean"] = df["resolution"].apply(clean_text)

    df["error_codes"] = df["description"].apply(extract_error_codes).apply(json.dumps)

    df["quality_score"] = df.apply(compute_text_quality_score, axis=1)

    df["embedding_text"] = df.apply(
        lambda r: f"Title: {r['title_clean']}\n"
                  f"Description: {r['description_clean']}\n"
                  f"Category: {r.get('category', '')}\n"
                  f"Component: {r.get('component', '')}\n"
                  f"Severity: {r.get('severity', '')}",
        axis=1
    )

    resolved = df[df["resolution_clean"].str.len() > 5].copy()
    resolved["knowledge_text"] = resolved.apply(
        lambda r: f"Problem: {r['title_clean']}\n"
                  f"Description: {r['description_clean']}\n"
                  f"Resolution: {r['resolution_clean']}\n"
                  f"Component: {r.get('component', '')}\n"
                  f"Category: {r.get('category', '')}",
        axis=1
    )

    df["source"] = "synthetic"
    return df


def process_jira_tickets():
    path = RAW_DIR / "jira_issues.csv"
    if not path.exists():
        print("  ⚠ jira_issues.csv not found. Run: python scripts/fetch_jira.py")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"  Loaded Jira: {len(df)} issues")

    df = df.rename(columns={
        "key": "ticket_id",
        "summary": "title",
        "description": "description",
        "issue_type": "issue_type",
        "priority": "severity",
        "status": "status",
        "resolution": "resolution",
        "components": "component",
    })

    df["title_clean"] = df["title"].apply(clean_text)
    df["description_clean"] = df["description"].apply(clean_text)
    df["resolution_clean"] = df["resolution"].fillna("").apply(clean_text)
    df["error_codes"] = df["description"].apply(extract_error_codes).apply(json.dumps)
    df["quality_score"] = df.apply(compute_text_quality_score, axis=1)

    type_to_category = {
        "Bug": "Software", "Improvement": "Software", "New Feature": "Software",
        "Task": "Software", "Sub-task": "Software", "Wish": "Software",
    }
    df["category"] = df["issue_type"].map(type_to_category).fillna("Software")

    df["embedding_text"] = df.apply(
        lambda r: f"Title: {r['title_clean']}\n"
                  f"Description: {r['description_clean'][:500]}\n"
                  f"Component: {r.get('component', '')}\n"
                  f"Priority: {r.get('severity', '')}",
        axis=1
    )

    df["source"] = "jira"
    return df


def process_github_issues():
    path = RAW_DIR / "github_issues.csv"
    if not path.exists():
        print("  ⚠ github_issues.csv not found. Run: python scripts/fetch_github_issues.py")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"  Loaded GitHub: {len(df)} issues")

    df = df.rename(columns={
        "number": "ticket_id",
        "title": "title",
        "body": "description",
        "labels": "labels",
    })
    df["ticket_id"] = df["repo"].apply(lambda x: x.split("/")[-1]) + "-" + df["ticket_id"].astype(str)

    df["title_clean"] = df["title"].apply(clean_text)
    df["description_clean"] = df["description"].apply(clean_text)
    df["error_codes"] = df["description"].apply(extract_error_codes).apply(json.dumps)

    def infer_severity(labels):
        if not isinstance(labels, str):
            return "Medium"
        labels_lower = labels.lower()
        if any(w in labels_lower for w in ["critical", "p0", "urgent", "blocker"]):
            return "Critical"
        if any(w in labels_lower for w in ["high", "p1", "important"]):
            return "High"
        if any(w in labels_lower for w in ["low", "p3", "minor", "trivial"]):
            return "Low"
        return "Medium"

    df["severity"] = df["labels"].apply(infer_severity)
    df["category"] = "Software"
    df["component"] = df.get("repo", "").apply(lambda x: x.split("/")[-1] if isinstance(x, str) else "")
    df["resolution"] = ""
    df["resolution_clean"] = ""
    df["quality_score"] = df.apply(compute_text_quality_score, axis=1)

    df["embedding_text"] = df.apply(
        lambda r: f"Title: {r['title_clean']}\n"
                  f"Description: {r['description_clean'][:500]}\n"
                  f"Labels: {r.get('labels', '')}\n"
                  f"Repo: {r.get('component', '')}",
        axis=1
    )

    df["source"] = "github"
    return df


# Maps our standard names → common column names we look for in unknown CSVs.
_COL_CANDIDATES = {
    "ticket_id":  ["ticket_id", "id", "key", "number", "issue_id", "TicketId", "sys_id", "incident_number"],
    "title":      ["title", "summary", "subject", "short_description", "name", "headline"],
    "description":["description", "body", "details", "long_description", "text", "notes", "comments"],
    "category":   ["category", "type", "issue_type", "call_type", "incident_type"],
    "component":  ["component", "service", "module", "service_name", "configuration_item", "cmdb_ci"],
    "assignment_group": ["assignment_group", "assigned_group", "support_group", "resolver_group", "team", "queue", "owner_group"],
    "department": ["department", "dept", "business_unit", "support_department", "assigned_department"],
    "severity":   ["severity", "priority", "urgency", "impact", "Priority"],
    "status":     ["status", "state", "incident_state"],
    "resolution": ["resolution", "close_notes", "resolve_notes", "solution", "fix", "resolution_notes"],
    "issue_type": ["issue_type", "type", "category", "call_type"],
}


def _auto_map_columns(df_cols: list, user_map: dict) -> dict:
    """Merge --column-map overrides with the table above."""
    mapping = {}
    col_set = set(df_cols)
    for std, candidates in _COL_CANDIDATES.items():
        if std in user_map and user_map[std] in col_set:
            mapping[std] = user_map[std]
        else:
            for c in candidates:
                if c in col_set:
                    mapping[std] = c
                    break
    return mapping


def process_csv_source(source_dir: Path, source_name: str, user_col_map: dict | None = None) -> pd.DataFrame:
    """Load all CSVs in a folder, map columns, clean, return the usual ticket schema."""
    csv_files = sorted(source_dir.glob("*.csv"))
    if not csv_files:
        print(f"  ⚠ No CSV files found in {source_dir}")
        return pd.DataFrame()

    frames = []
    for p in csv_files:
        try:
            frames.append(pd.read_csv(p, low_memory=False))
            print(f"  Loaded {p.name}: {len(frames[-1])} rows")
        except Exception as e:
            print(f"  ⚠ Could not read {p.name}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    print(f"  Total {source_name}: {len(df)} rows across {len(csv_files)} file(s)")

    col_map = _auto_map_columns(list(df.columns), user_col_map or {})
    print(f"  Column mapping: {col_map}")

    rename = {v: k for k, v in col_map.items() if v != k and v in df.columns}
    df = df.rename(columns=rename)

    for col in ["ticket_id", "title", "description", "category", "component",
                "assignment_group", "department", "severity", "status", "resolution", "issue_type"]:
        if col not in df.columns:
            df[col] = ""

    for col in ["title", "description", "resolution", "component", "assignment_group",
                "department", "severity", "status", "category", "issue_type", "ticket_id"]:
        df[col] = df[col].fillna("").astype(str)

    df["raw_assignment_group"] = df["assignment_group"]
    df["raw_department"] = df["department"]
    df["mapped_department"] = df.apply(
        lambda r: map_to_department(
            r.get("department", ""),
            r.get("assignment_group", ""),
            r.get("component", ""),
            r.get("category", ""),
            r.get("title", ""),
            r.get("description", ""),
        ),
        axis=1,
    )
    df["department"] = df["mapped_department"]

    df["title_clean"]       = df["title"].apply(clean_text)
    df["description_clean"] = df["description"].apply(clean_text)
    df["resolution_clean"]  = df["resolution"].apply(clean_text)
    df["error_codes"]       = df["description"].apply(extract_error_codes).apply(json.dumps)
    df["quality_score"]     = df.apply(compute_text_quality_score, axis=1)

    # Include resolution when we have it so “how it was fixed” can match queries.
    def _make_embedding_text(r):
        parts = [
            f"Title: {r['title_clean']}",
            f"Description: {r['description_clean'][:600]}",
        ]
        if r.get("category"):
            parts.append(f"Category: {r['category']}")
        if r.get("component"):
            parts.append(f"Component: {r['component']}")
        if r.get("department"):
            parts.append(f"Department: {r['department']}")
        if r.get("assignment_group"):
            parts.append(f"Assignment Group: {r['assignment_group']}")
        if r.get("severity"):
            parts.append(f"Severity: {r['severity']}")
        if r.get("resolution_clean") and len(str(r["resolution_clean"])) > 5:
            parts.append(f"Resolution: {r['resolution_clean'][:300]}")
        return "\n".join(parts)

    df["embedding_text"] = df.apply(_make_embedding_text, axis=1)
    df["source"] = source_name
    return df


def process_major_its_dataset(path: Path = MAJOR_TICKETS_PATH) -> pd.DataFrame:
    """Main bucket: the big ITS export (e.g. ~80k rows)."""
    if not path.exists():
        print(f"  ⚠ Major ITS dataset not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded major ITS dataset: {len(df)} tickets from {path.name}")

    if "external_source" not in df.columns:
        df["external_source"] = df.get("source", "")
    if "source_system" not in df.columns:
        df["source_system"] = df.get("external_source", df.get("source", ""))

    for col in [
        "ticket_id", "title", "title_clean", "description", "description_clean",
        "category", "component", "issue_type", "severity", "status",
        "resolution", "resolution_clean", "error_codes", "embedding_text",
        "language", "tags", "import_id", "external_record_id", "external_source",
        "source_system",
    ]:
        if col not in df.columns:
            df[col] = ""

    for col in [
        "ticket_id", "title", "description", "category", "component", "issue_type",
        "severity", "status", "resolution", "language", "tags", "external_source",
        "source_system",
    ]:
        df[col] = df[col].fillna("").astype(str)

    df["title_clean"] = df["title"].apply(clean_text)
    df["description_clean"] = df["description"].apply(clean_text)
    df["resolution_clean"] = df["resolution"].apply(clean_text)
    df["error_codes"] = df["description"].apply(extract_error_codes).apply(json.dumps)
    df["quality_score"] = df.apply(compute_text_quality_score, axis=1)

    df["assignment_group"] = ""
    df["raw_assignment_group"] = ""
    df["raw_department"] = df["category"]
    df["mapped_department"] = df.apply(
        lambda r: map_to_department(
            r.get("category", ""),
            r.get("component", ""),
            r.get("tags", ""),
            r.get("title", ""),
            r.get("description", ""),
        ),
        axis=1,
    )
    df["department"] = df["mapped_department"]

    def _make_embedding_text(r):
        parts = [
            f"Title: {r['title_clean']}",
            f"Description: {str(r['description_clean'])[:700]}",
            f"Category: {r.get('category', '')}",
            f"Component: {r.get('component', '')}",
            f"Department: {r.get('department', '')}",
            f"Severity: {r.get('severity', '')}",
        ]
        if r.get("tags"):
            parts.append(f"Tags: {r.get('tags', '')}")
        if r.get("resolution_clean") and len(str(r["resolution_clean"])) > 5:
            parts.append(f"Resolution: {str(r['resolution_clean'])[:400]}")
        return "\n".join(parts)

    df["embedding_text"] = df.apply(_make_embedding_text, axis=1)
    df["source"] = "major_its"
    return df


def process_knowledge_base():
    """Index every *.md under data/knowledge_base (recursive)."""
    kb_dir = Path("./data/knowledge_base")
    if not kb_dir.exists():
        print("  ⚠ Knowledge base not found.")
        return []

    documents = []
    for filepath in sorted(kb_dir.rglob("*.md")):
        with open(filepath, "r") as f:
            content = f.read()

        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else filepath.stem

        doc_type = "runbook" if "runbook" in filepath.stem else "kb_article"

        documents.append({
            "doc_id": filepath.stem,
            "title": title,
            "content": content,
            "doc_type": doc_type,
            "source_file": filepath.name,
            "char_count": len(content),
            "word_count": len(content.split()),
        })
        print(f"  Processed: {filepath.name} ({len(content)} chars)")

    df_docs = pd.DataFrame(documents)
    out_path = PROCESSED_DIR / "knowledge_base_index.csv"
    df_docs.to_csv(out_path, index=False)
    print(f"  KB Index: {len(df_docs)} documents → {out_path}")

    return documents


STANDARD_COLS = [
    "ticket_id", "title", "title_clean", "description", "description_clean",
    "category", "component", "assignment_group", "raw_assignment_group",
    "raw_department", "department", "mapped_department", "issue_type", "severity", "status",
    "resolution", "resolution_clean", "error_codes", "quality_score",
    "embedding_text", "source", "source_system", "external_source",
    "external_record_id", "import_id", "language", "tags",
]


def merge_all_tickets(
    extra_sources: list | None = None,
    append: bool = False,
    min_quality: float = 0.0,
    include_major: bool = True,
    major_path: Path | None = None,
) -> pd.DataFrame | None:
    """Stack sources into one table; optional append mode skips existing ticket_id+source pairs."""
    print("\n" + "=" * 50)
    print("Merging all ticket sources...")
    print("=" * 50)

    frames = []

    if not append:
        # Full rebuild — process every known source
        if include_major:
            print("\n[0/4] Processing major ITS tickets...")
            df_major = process_major_its_dataset(major_path or MAJOR_TICKETS_PATH)
            if len(df_major) > 0:
                frames.append(df_major)

        print("\n[1/4] Processing synthetic tickets...")
        df_synth = process_synthetic_tickets()
        if len(df_synth) > 0:
            frames.append(df_synth)

        print("\n[2/4] Processing Jira tickets...")
        df_jira = process_jira_tickets()
        if len(df_jira) > 0:
            frames.append(df_jira)

        print("\n[3/4] Processing GitHub issues...")
        df_github = process_github_issues()
        if len(df_github) > 0:
            frames.append(df_github)

    if extra_sources:
        for df_extra in extra_sources:
            if len(df_extra) > 0:
                frames.append(df_extra)

    if not frames:
        print("\n❌ No ticket data found!")
        return None

    for df in frames:
        for col in ["assignment_group", "raw_assignment_group", "raw_department"]:
            if col not in df.columns:
                df[col] = ""
        if "department" not in df.columns:
            df["department"] = df.apply(
                lambda r: map_to_department(
                    r.get("component", ""),
                    r.get("category", ""),
                    r.get("title", ""),
                    r.get("description", ""),
                ),
                axis=1,
            )
        if "mapped_department" not in df.columns:
            df["mapped_department"] = df["department"]
        for col in STANDARD_COLS:
            if col not in df.columns:
                df[col] = ""

    df_new = pd.concat([df[STANDARD_COLS] for df in frames], ignore_index=True)

    if min_quality > 0.0:
        before = len(df_new)
        df_new = df_new[df_new["quality_score"] >= min_quality].reset_index(drop=True)
        print(f"  Quality filter (>= {min_quality}): {before} → {len(df_new)} tickets")

    df_new = df_new[df_new["title_clean"].str.len() > 3].reset_index(drop=True)
    df_new = df_new[df_new["embedding_text"].str.len() > 10].reset_index(drop=True)

    if append:
        out_csv = PROCESSED_DIR / "all_tickets.csv"
        if out_csv.exists():
            df_existing = pd.read_csv(out_csv).fillna("")
            existing_keys = set(
                zip(df_existing["ticket_id"].astype(str), df_existing["source"].astype(str))
            )
            before = len(df_new)
            df_new = df_new[
                ~df_new.apply(
                    lambda r: (str(r["ticket_id"]), str(r["source"])) in existing_keys, axis=1
                )
            ].reset_index(drop=True)
            print(f"  Dedup against existing: {before} → {len(df_new)} truly new tickets")

            max_existing = len(df_existing)
            df_new["unified_id"] = [f"U-{max_existing + i:06d}" for i in range(len(df_new))]

            df_all = pd.concat([df_existing, df_new], ignore_index=True)
            print(f"  Appended {len(df_new)} rows; total now {len(df_all)}")
        else:
            print("  --append requested but no existing file found; doing full write.")
            df_new["unified_id"] = [f"U-{i:06d}" for i in range(len(df_new))]
            df_all = df_new
    else:
        before = len(df_new)
        df_new = df_new.drop_duplicates(subset=["embedding_text"]).reset_index(drop=True)
        if len(df_new) < before:
            print(f"  Removed {before - len(df_new)} exact-duplicate embedding texts")
        df_new["unified_id"] = [f"U-{i:06d}" for i in range(len(df_new))]
        df_all = df_new

    out_csv = PROCESSED_DIR / "all_tickets.csv"
    df_all.to_csv(out_csv, index=False)

    out_json = PROCESSED_DIR / "all_tickets_clean.json"
    with open(out_json, "w") as f:
        json.dump(df_all.to_dict(orient="records"), f, indent=2, default=str)

    print(f"\n  Unified dataset: {len(df_all)} tickets")
    print(f"  Saved: {out_csv}")
    print(f"  Saved: {out_json}")

    mapping_report = {
        "final_departments": DEPARTMENTS,
        "department_counts": df_all["department"].fillna("").value_counts().to_dict(),
        "raw_assignment_group_top_30": df_all["raw_assignment_group"].fillna("").value_counts().head(30).to_dict()
        if "raw_assignment_group" in df_all.columns else {},
        "raw_department_top_30": df_all["raw_department"].fillna("").value_counts().head(30).to_dict()
        if "raw_department" in df_all.columns else {},
        "category_top_30": df_all["category"].fillna("").value_counts().head(30).to_dict(),
        "component_top_30": df_all["component"].fillna("").value_counts().head(30).to_dict(),
    }
    mapping_path = PROCESSED_DIR / "department_mapping_report.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping_report, f, indent=2)
    print(f"  Department mapping report: {mapping_path}")
    return df_all


def generate_stats(df_all):
    """Write data_stats.json and print a quick summary."""
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)

    stats = {
        "total_tickets": len(df_all),
        "sources": df_all["source"].value_counts().to_dict(),
        "categories": df_all["category"].value_counts().to_dict(),
        "departments": df_all["department"].value_counts().to_dict()
        if "department" in df_all.columns else {},
        "source_systems": df_all["source_system"].value_counts().head(30).to_dict()
        if "source_system" in df_all.columns else {},
        "languages": df_all["language"].value_counts().to_dict()
        if "language" in df_all.columns else {},
        "severities": df_all["severity"].value_counts().to_dict(),
        "statuses": df_all["status"].value_counts().to_dict(),
        "avg_title_length": round(df_all["title_clean"].str.len().mean(), 1),
        "avg_description_length": round(df_all["description_clean"].str.len().mean(), 1),
        "avg_quality_score": round(df_all["quality_score"].mean(), 3),
        "tickets_with_resolution": int((df_all["resolution_clean"].str.len() > 5).sum()),
        "tickets_with_error_codes": int((df_all["error_codes"].str.len() > 2).sum()),
        "embedding_text_avg_length": round(df_all["embedding_text"].str.len().mean(), 1),
    }

    for key, val in stats.items():
        if isinstance(val, dict):
            print(f"\n  {key}:")
            for k, v in val.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {val}")

    stats_path = PROCESSED_DIR / "data_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved: {stats_path}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ITS RAG - Data Preprocessing")
    parser.add_argument(
        "--new-source", metavar="DIR",
        help="Path to a folder of CSV files to add as a new source. "
             "Columns are auto-detected; override with --column-map.",
    )
    parser.add_argument(
        "--dataset-name", metavar="NAME", default="custom",
        help="Label for the new source (stored in the 'source' column). "
             "Default: custom",
    )
    parser.add_argument(
        "--column-map", metavar="JSON", default="{}",
        help='JSON dict mapping standard column names to actual CSV column names. '
             'Example: \'{"title":"short_desc","resolution":"close_notes"}\'',
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append new rows to existing all_tickets.csv instead of rebuilding. "
             "Existing rows (matched by ticket_id + source) are skipped.",
    )
    parser.add_argument(
        "--min-quality", type=float, default=0.0, metavar="SCORE",
        help="Drop tickets with quality_score below this value (0.0–1.0). Default: 0.0",
    )
    parser.add_argument(
        "--no-kb", action="store_true",
        help="Skip knowledge base processing (useful when only adding ticket data).",
    )
    parser.add_argument(
        "--skip-major-data", action="store_true",
        help="Do not include the major ITS CSV (its_ticket80.csv / its_tickets_80k.csv) during a full rebuild.",
    )
    parser.add_argument(
        "--major-path",
        default=str(default_major_its_csv_path()),
        help="Path to the major ITS tickets CSV. Default: first existing of "
        "data/processed/its_ticket80.csv, data/processed/its_tickets_80k.csv",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ITS RAG - Step 2: Data Preprocessing")
    print("=" * 60)

    try:
        user_col_map = json.loads(args.column_map)
    except json.JSONDecodeError as e:
        print(f"❌ --column-map is not valid JSON: {e}")
        raise SystemExit(1)

    extra_sources = []
    if args.new_source:
        src_path = Path(args.new_source)
        if not src_path.is_dir():
            print(f"❌ --new-source path does not exist or is not a directory: {src_path}")
            raise SystemExit(1)
        print(f"\n[NEW SOURCE] Processing '{args.dataset_name}' from {src_path}...")
        df_new_src = process_csv_source(src_path, args.dataset_name, user_col_map)
        if len(df_new_src) > 0:
            extra_sources.append(df_new_src)
        else:
            print("  ⚠ No usable rows found in new source.")

    df_all = merge_all_tickets(
        extra_sources=extra_sources if extra_sources else None,
        append=args.append,
        min_quality=args.min_quality,
        include_major=not args.skip_major_data,
        major_path=Path(args.major_path),
    )

    if df_all is not None:
        if not args.no_kb:
            print("\n[KB] Processing Knowledge Base...")
            kb_docs = process_knowledge_base()
        else:
            kb_docs = []

        stats = generate_stats(df_all)

        print("\n" + "=" * 60)
        print("✅ Preprocessing complete!")
        print(f"  Tickets: {len(df_all)} → data/processed/all_tickets.csv")
        if not args.no_kb:
            print(f"  KB Docs: {len(kb_docs)} → data/processed/knowledge_base_index.csv")
        print(f"  Stats:   → data/processed/data_stats.json")
        print("\nNext step: python 03_build_vector_store.py [--append | --reset]")
        print("=" * 60)

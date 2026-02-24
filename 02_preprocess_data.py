"""
ITS RAG - Step 2: Data Preprocessing
Cleans, normalizes, and prepares all datasets for embedding and indexing.

Outputs:
  - processed/all_tickets.csv          (unified ticket dataset)
  - processed/all_tickets_clean.json   (JSON for LLM consumption)
  - processed/duplicate_pairs.csv      (ground truth for dedup eval)
  - processed/data_stats.json          (dataset statistics)

Usage:
  python 02_preprocess_data.py
"""

import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

RAW_DIR = Path("./data/raw")
PROCESSED_DIR = Path("./data/processed")
KB_DIR = Path("./data/knowledge_base")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Text Cleaning Utilities
# ──────────────────────────────────────────────
def clean_text(text):
    """Clean and normalize text for embedding."""
    if not isinstance(text, str) or not text.strip():
        return ""
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '[URL]', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
    # Remove file paths
    text = re.sub(r'[A-Za-z]:\\[\w\\\.]+', '[PATH]', text)
    text = re.sub(r'/[\w/\.]+', '[PATH]', text)
    # Remove hex addresses and memory dumps
    text = re.sub(r'0x[0-9a-fA-F]+', '[HEX]', text)
    # Remove excessive whitespace but keep newlines for structure
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()


def extract_error_codes(text):
    """Extract error codes and identifiers from text."""
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
    """Score ticket quality 0-1 based on completeness."""
    score = 0
    if row.get("title") and len(str(row["title"])) > 5:
        score += 0.2
    if row.get("description") and len(str(row["description"])) > 20:
        score += 0.3
    if row.get("component") and str(row["component"]).strip():
        score += 0.15
    if row.get("severity") and str(row["severity"]).strip():
        score += 0.1
    if row.get("environment") and str(row["environment"]).strip():
        score += 0.1
    if row.get("steps_to_reproduce") and len(str(row["steps_to_reproduce"])) > 10:
        score += 0.15
    return round(score, 2)


# ──────────────────────────────────────────────
# Process Each Data Source
# ──────────────────────────────────────────────
def process_synthetic_tickets():
    """Process synthetic tickets."""
    path = PROCESSED_DIR / "synthetic_tickets.csv"
    if not path.exists():
        print("  ⚠ synthetic_tickets.csv not found. Run 01_download_data.py first.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"  Loaded synthetic: {len(df)} tickets")

    # Clean text fields
    df["title_clean"] = df["title"].apply(clean_text)
    df["description_clean"] = df["description"].apply(clean_text)
    df["resolution_clean"] = df["resolution"].apply(clean_text)

    # Extract error codes
    df["error_codes"] = df["description"].apply(extract_error_codes).apply(json.dumps)

    # Quality score
    df["quality_score"] = df.apply(compute_text_quality_score, axis=1)

    # Create combined text for embedding
    df["embedding_text"] = df.apply(
        lambda r: f"Title: {r['title_clean']}\n"
                  f"Description: {r['description_clean']}\n"
                  f"Category: {r.get('category', '')}\n"
                  f"Component: {r.get('component', '')}\n"
                  f"Severity: {r.get('severity', '')}",
        axis=1
    )

    # Create resolution text for RAG knowledge
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
    """Process Apache Jira tickets."""
    path = RAW_DIR / "jira_issues.csv"
    if not path.exists():
        print("  ⚠ jira_issues.csv not found. Run: python scripts/fetch_jira.py")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"  Loaded Jira: {len(df)} issues")

    # Standardize columns
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

    # Clean
    df["title_clean"] = df["title"].apply(clean_text)
    df["description_clean"] = df["description"].apply(clean_text)
    df["resolution_clean"] = df["resolution"].fillna("").apply(clean_text)
    df["error_codes"] = df["description"].apply(extract_error_codes).apply(json.dumps)
    df["quality_score"] = df.apply(compute_text_quality_score, axis=1)

    # Category mapping
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
    """Process GitHub Issues."""
    path = RAW_DIR / "github_issues.csv"
    if not path.exists():
        print("  ⚠ github_issues.csv not found. Run: python scripts/fetch_github_issues.py")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"  Loaded GitHub: {len(df)} issues")

    # Standardize
    df = df.rename(columns={
        "number": "ticket_id",
        "title": "title",
        "body": "description",
        "labels": "labels",
    })
    df["ticket_id"] = df["repo"].apply(lambda x: x.split("/")[-1]) + "-" + df["ticket_id"].astype(str)

    # Clean
    df["title_clean"] = df["title"].apply(clean_text)
    df["description_clean"] = df["description"].apply(clean_text)
    df["error_codes"] = df["description"].apply(extract_error_codes).apply(json.dumps)

    # Infer severity from labels
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


# ──────────────────────────────────────────────
# Process Knowledge Base Documents
# ──────────────────────────────────────────────
def process_knowledge_base():
    """Process KB documents into chunks ready for RAG."""
    kb_dir = Path("./data/knowledge_base")
    if not kb_dir.exists():
        print("  ⚠ Knowledge base not found. Run 01_download_data.py --generate-kb")
        return []

    documents = []
    for filepath in sorted(kb_dir.glob("*.md")):
        with open(filepath, "r") as f:
            content = f.read()

        # Extract title from first heading
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else filepath.stem

        # Determine document type
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

    # Save document index
    df_docs = pd.DataFrame(documents)
    out_path = PROCESSED_DIR / "knowledge_base_index.csv"
    df_docs.to_csv(out_path, index=False)
    print(f"  KB Index: {len(df_docs)} documents → {out_path}")

    return documents


# ──────────────────────────────────────────────
# Merge & Create Unified Dataset
# ──────────────────────────────────────────────
def merge_all_tickets():
    """Merge all ticket sources into a unified dataset."""
    print("\n" + "=" * 50)
    print("Merging all ticket sources...")
    print("=" * 50)

    frames = []

    # Process each source
    print("\n[1/3] Processing synthetic tickets...")
    df_synth = process_synthetic_tickets()
    if len(df_synth) > 0:
        frames.append(df_synth)

    print("\n[2/3] Processing Jira tickets...")
    df_jira = process_jira_tickets()
    if len(df_jira) > 0:
        frames.append(df_jira)

    print("\n[3/3] Processing GitHub issues...")
    df_github = process_github_issues()
    if len(df_github) > 0:
        frames.append(df_github)

    if not frames:
        print("\n❌ No ticket data found! Run 01_download_data.py first.")
        return None

    # Standardize columns across all sources
    standard_cols = [
        "ticket_id", "title", "title_clean", "description", "description_clean",
        "category", "component", "issue_type", "severity", "status",
        "resolution", "resolution_clean", "error_codes", "quality_score",
        "embedding_text", "source"
    ]

    for df in frames:
        for col in standard_cols:
            if col not in df.columns:
                df[col] = ""

    # Merge
    df_all = pd.concat([df[standard_cols] for df in frames], ignore_index=True)

    # Remove empty/garbage tickets
    df_all = df_all[df_all["title_clean"].str.len() > 3].reset_index(drop=True)
    df_all = df_all[df_all["embedding_text"].str.len() > 10].reset_index(drop=True)

    # Assign unique IDs
    df_all["unified_id"] = [f"U-{i:06d}" for i in range(len(df_all))]

    # Save
    out_csv = PROCESSED_DIR / "all_tickets.csv"
    df_all.to_csv(out_csv, index=False)

    out_json = PROCESSED_DIR / "all_tickets_clean.json"
    records = df_all.to_dict(orient="records")
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2, default=str)

    print(f"\n  Unified dataset: {len(df_all)} tickets")
    print(f"  Saved: {out_csv}")
    print(f"  Saved: {out_json}")

    return df_all


# ──────────────────────────────────────────────
# Generate Statistics Report
# ──────────────────────────────────────────────
def generate_stats(df_all):
    """Generate comprehensive dataset statistics."""
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)

    stats = {
        "total_tickets": len(df_all),
        "sources": df_all["source"].value_counts().to_dict(),
        "categories": df_all["category"].value_counts().to_dict(),
        "severities": df_all["severity"].value_counts().to_dict(),
        "statuses": df_all["status"].value_counts().to_dict(),
        "avg_title_length": round(df_all["title_clean"].str.len().mean(), 1),
        "avg_description_length": round(df_all["description_clean"].str.len().mean(), 1),
        "avg_quality_score": round(df_all["quality_score"].mean(), 3),
        "tickets_with_resolution": int((df_all["resolution_clean"].str.len() > 5).sum()),
        "tickets_with_error_codes": int((df_all["error_codes"].str.len() > 2).sum()),
        "embedding_text_avg_length": round(df_all["embedding_text"].str.len().mean(), 1),
    }

    # Print stats
    for key, val in stats.items():
        if isinstance(val, dict):
            print(f"\n  {key}:")
            for k, v in val.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {val}")

    # Save stats
    stats_path = PROCESSED_DIR / "data_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Stats saved: {stats_path}")

    return stats


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("ITS RAG - Step 2: Data Preprocessing")
    print("=" * 60)

    # Process tickets
    df_all = merge_all_tickets()

    if df_all is not None:
        # Process knowledge base
        print("\n[KB] Processing Knowledge Base...")
        kb_docs = process_knowledge_base()

        # Generate stats
        stats = generate_stats(df_all)

        print("\n" + "=" * 60)
        print("✅ Preprocessing complete!")
        print(f"  Tickets: {len(df_all)} → data/processed/all_tickets.csv")
        print(f"  KB Docs: {len(kb_docs)} → data/processed/knowledge_base_index.csv")
        print(f"  Stats:   → data/processed/data_stats.json")
        print("\nNext step: python 03_build_vector_store.py")
        print("=" * 60)

"""
Fetch Apache JIRA Issues via REST API (No auth needed for public projects)
"""
import requests
import pandas as pd
import json
import time
from pathlib import Path

JIRA_URL = "https://issues.apache.org/jira/rest/api/2/search"
OUTPUT_DIR = Path("./data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROJECTS = ["SPARK", "KAFKA", "HADOOP", "CASSANDRA", "FLINK"]
MAX_PER_PROJECT = 500  # Adjust up to 1000

def fetch_project_issues(project, max_results=500):
    """Fetch issues from an Apache JIRA project."""
    all_issues = []
    start_at = 0
    batch_size = 50  # JIRA API max per request

    print(f"  Fetching {project}...")
    while start_at < max_results:
        params = {
            "jql": f"project = {project} ORDER BY created DESC",
            "startAt": start_at,
            "maxResults": min(batch_size, max_results - start_at),
            "fields": "summary,description,issuetype,priority,status,resolution,"
                      "components,labels,created,updated,resolutiondate,comment,"
                      "assignee,reporter"
        }

        try:
            resp = requests.get(JIRA_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            issues = data.get("issues", [])
            if not issues:
                break

            for issue in issues:
                fields = issue["fields"]
                comments_text = ""
                if fields.get("comment") and fields["comment"].get("comments"):
                    comments_text = " | ".join([
                        c.get("body", "")[:500] for c in fields["comment"]["comments"][:5]
                    ])

                all_issues.append({
                    "key": issue["key"],
                    "project": project,
                    "summary": fields.get("summary", ""),
                    "description": (fields.get("description") or "")[:3000],
                    "issue_type": fields.get("issuetype", {}).get("name", ""),
                    "priority": fields.get("priority", {}).get("name", ""),
                    "status": fields.get("status", {}).get("name", ""),
                    "resolution": (fields.get("resolution") or {}).get("name", ""),
                    "components": ", ".join([c["name"] for c in (fields.get("components") or [])]),
                    "labels": ", ".join(fields.get("labels") or []),
                    "created": fields.get("created", ""),
                    "updated": fields.get("updated", ""),
                    "resolution_date": fields.get("resolutiondate", ""),
                    "comments": comments_text,
                    "assignee": (fields.get("assignee") or {}).get("displayName", ""),
                    "reporter": (fields.get("reporter") or {}).get("displayName", ""),
                })

            start_at += len(issues)
            print(f"    {project}: {start_at} issues fetched...")
            time.sleep(0.5)  # Rate limit

        except Exception as e:
            print(f"    Error at {start_at}: {e}")
            break

    return all_issues

if __name__ == "__main__":
    all_data = []
    for project in PROJECTS:
        issues = fetch_project_issues(project, MAX_PER_PROJECT)
        all_data.extend(issues)
        print(f"  {project}: {len(issues)} issues")

    df = pd.DataFrame(all_data)
    out_path = OUTPUT_DIR / "jira_issues.csv"
    df.to_csv(out_path, index=False)
    print(f"\nTotal: {len(df)} issues saved to {out_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample:\n{df.head()}")

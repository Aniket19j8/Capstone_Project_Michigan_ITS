"""
Fetch GitHub Issues from high-volume repos via REST API
Set GITHUB_TOKEN env var for higher rate limits (5000/hr vs 60/hr)
"""
import os
import requests
import pandas as pd
import time
from pathlib import Path

OUTPUT_DIR = Path("./data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOKEN = os.getenv("GITHUB_TOKEN", "")
HEADERS = {"Authorization": f"token {TOKEN}"} if TOKEN else {}

REPOS = [
    "microsoft/vscode",
    "tensorflow/tensorflow",
    "pytorch/pytorch",
    "kubernetes/kubernetes",
    "facebook/react",
]
MAX_PER_REPO = 300

def fetch_repo_issues(repo, max_issues=300):
    """Fetch closed issues with labels and comments."""
    all_issues = []
    page = 1
    per_page = 100

    print(f"  Fetching {repo}...")
    while len(all_issues) < max_issues:
        url = f"https://api.github.com/repos/{repo}/issues"
        params = {
            "state": "closed",
            "per_page": per_page,
            "page": page,
            "sort": "updated",
            "direction": "desc"
        }

        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
            if resp.status_code == 403:
                print(f"    Rate limited. Set GITHUB_TOKEN env var.")
                break
            resp.raise_for_status()
            issues = resp.json()
            if not issues:
                break

            for issue in issues:
                if issue.get("pull_request"):  # Skip PRs
                    continue
                all_issues.append({
                    "repo": repo,
                    "number": issue["number"],
                    "title": issue["title"],
                    "body": (issue.get("body") or "")[:3000],
                    "labels": ", ".join([l["name"] for l in issue.get("labels", [])]),
                    "state": issue["state"],
                    "created_at": issue["created_at"],
                    "closed_at": issue.get("closed_at", ""),
                    "comments_count": issue.get("comments", 0),
                    "user": issue.get("user", {}).get("login", ""),
                    "url": issue["html_url"],
                })

            page += 1
            print(f"    {repo}: {len(all_issues)} issues...")
            time.sleep(1)

        except Exception as e:
            print(f"    Error: {e}")
            break

    return all_issues[:max_issues]

if __name__ == "__main__":
    all_data = []
    for repo in REPOS:
        issues = fetch_repo_issues(repo, MAX_PER_REPO)
        all_data.extend(issues)
        print(f"  {repo}: {len(issues)} issues")

    df = pd.DataFrame(all_data)
    out_path = OUTPUT_DIR / "github_issues.csv"
    df.to_csv(out_path, index=False)
    print(f"\nTotal: {len(df)} issues saved to {out_path}")

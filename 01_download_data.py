"""
Step 1 — get data on disk without surprising network calls.

We stopped auto-fetching public datasets on a plain `python 01_download_data.py` run.
Reason: capstone machines / air-gapped setups / rate limits. You either (a) drop files in
place, or (b) opt in with --with-public-datasets.

What we actually use in the story:
  • The big ticket CSV (~80k rows) — download from class/Google Drive manually, drop as
    data/processed/its_tickets_80k.csv or its_ticket80.csv. This script only *registers*
    it (manifest for step 02). No HTTP for that file.
  • Optional: precomputed Qwen3-0.6B vectors for the 80k rows (Drive link in README)
    — share is mostly for reproducing our embedding study; 03 still embeds by default unless you wire in a loader yourself.
  • Synthetic tickets + stub KB markdown — generated here if you have no real export yet.

Public mirrors (GitBugs, Jira, GitHub, Quora) are extras for experiments; pip install
datasets if you use --with-public-datasets.

Examples:
  python 01_download_data.py                      # local: register major CSV + synthetic + KB
  python 01_download_data.py --register-major-data
  python 01_download_data.py --generate-synthetic --synthetic-count 300
  python 01_download_data.py --with-public-datasets   # hits HuggingFace / APIs
"""

import os
import json
import argparse
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("./data/raw")
PROCESSED_DIR = Path("./data/processed")
# two filenames floated in our class export — whichever you have is fine
_MAJOR_CANDIDATES = (
    PROCESSED_DIR / "its_ticket80.csv",
    PROCESSED_DIR / "its_tickets_80k.csv",
)


def default_major_its_csv_path() -> Path:
    # first match wins so teammates aren't fighting filenames
    for p in _MAJOR_CANDIDATES:
        if p.exists():
            return p
    return _MAJOR_CANDIDATES[-1]


MAJOR_TICKETS_PATH = default_major_its_csv_path()
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _configure_stdout_utf8():
    """Avoid UnicodeEncodeError on Windows terminals with legacy encodings."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# D0 — "major" corpus: you brought the file; we just sniff columns + row count
def register_major_its_dataset(path: Path = MAJOR_TICKETS_PATH):
    print("\n[D0] Registering major ITS tickets dataset...")
    if not path.exists():
        print(f"  ⚠ Major dataset not found at: {path}")
        print("  Place the CSV there, or pass a different path with --major-path.")
        return False

    # small read here — 02 does the heavy lift
    sample = pd.read_csv(path, nrows=500)
    columns = list(sample.columns)
    required = {"ticket_id", "title", "description", "category", "component", "status"}
    optional = {"resolution", "resolution_clean", "external_source", "language", "tags"}
    missing_required = sorted(required - set(columns))

    # cheap line count — don't load 80k rows twice
    with open(path, "rb") as f:
        row_count = max(sum(1 for _ in f) - 1, 0)

    profile = {
        "dataset_name": "major_its",
        "path": str(path),
        "filename": path.name,
        "row_count": row_count,
        "columns": columns,
        "missing_required_columns": missing_required,
        "available_optional_columns": sorted(optional & set(columns)),
        "sample_category_counts": sample.get("category", pd.Series(dtype=str)).fillna("").value_counts().head(20).to_dict(),
        "sample_component_counts": sample.get("component", pd.Series(dtype=str)).fillna("").value_counts().head(20).to_dict(),
        "sample_status_counts": sample.get("status", pd.Series(dtype=str)).fillna("").value_counts().head(20).to_dict(),
        "registered_for_step_02": True,
    }

    manifest_path = PROCESSED_DIR / "major_its_dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(f"  Rows: {row_count:,}")
    print(f"  Columns: {len(columns)}")
    if missing_required:
        print(f"  ⚠ Missing required columns: {missing_required}")
    else:
        print("  Required columns: OK")
    print(f"  Manifest: {manifest_path}")
    return not missing_required


# D1 — optional GitBugs (HF) — only when you ask for it
def download_gitbugs():
    print("\n[D1] GitBugs (HuggingFace)...")
    from datasets import load_dataset

    try:
        ds = load_dataset("logpai/GitBugs")
        print(f"  Loaded GitBugs: {ds}")
        for split_name, split_data in ds.items():
            df = split_data.to_pandas()
            out_path = RAW_DIR / f"gitbugs_{split_name}.csv"
            df.to_csv(out_path, index=False)
            print(f"  Saved {split_name}: {len(df)} rows → {out_path}")
        return True
    except Exception as e:
        print(f"  HuggingFace load failed: {e}")

    print("  (install datasets + check VPN if you need GitBugs manually)")
    return False


# D2 — optional Jira-shaped exports
def download_jira():
    print("\n[D2] Apache Jira paths...")

    try:
        from datasets import load_dataset

        candidates = [
            "apache-jira-bugs",
            "jirasec/jira-issues",
        ]

        for ds_name in candidates:
            try:
                ds = load_dataset(ds_name)
                df = ds["train"].to_pandas() if "train" in ds else list(ds.values())[0].to_pandas()
                out_path = RAW_DIR / "jira_issues.csv"
                df.to_csv(out_path, index=False)
                print(f"  Saved: {len(df)} issues → {out_path}")
                return True
            except:
                continue
    except:
        pass

    print("  scribbling scripts/fetch_jira.py — run that yourself if you want live Jira")
    generate_jira_api_script()
    return False


def generate_jira_api_script():
    """Spit out scripts/fetch_jira.py — we don't run live Jira from 01 by default."""
    script = '''"""
Pull Apache Jira issues (public projects, no auth).
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
MAX_PER_PROJECT = 500

def fetch_project_issues(project, max_results=500):
    all_issues = []
    start_at = 0
    batch_size = 50

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
            time.sleep(0.5)

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
    print(f"\\nTotal: {len(df)} issues saved to {out_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"\\nSample:\\n{df.head()}")
'''
    script_path = Path("./scripts/fetch_jira.py")
    script_path.parent.mkdir(parents=True, exist_ok=True)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"  Generated: {script_path}")


# D3 — GitHub: we only emit a helper script; token is on you
def download_github():
    print("\n[D3] GitHub helper script...")
    generate_github_script()
    print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║  Run the GitHub Issues fetcher:                            ║
  ║    python scripts/fetch_github_issues.py                   ║
  ║                                                            ║
  ║  Set GITHUB_TOKEN env var for higher rate limits:          ║
  ║    export GITHUB_TOKEN=ghp_your_token_here                 ║
  ║                                                            ║
  ║  Repos: vscode, tensorflow, pytorch, kubernetes, react     ║
  ╚══════════════════════════════════════════════════════════════╝
    """)


def generate_github_script():
    script = '''"""
Fetch issues from a few big repos. GITHUB_TOKEN = higher rate limit.
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
                if issue.get("pull_request"):
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
    print(f"\\nTotal: {len(df)} issues saved to {out_path}")
'''
    script_path = Path("./scripts/fetch_github_issues.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"  Generated: {script_path}")


# D4 — Quora duplicate pairs (optional sanity data)
def download_quora():
    # HF quora layout is annoyingly inconsistent; normalize to text1/text2
    print("\n[D4] Downloading Quora Duplicate Pairs...")
    try:
        from datasets import load_dataset
        ds = load_dataset("quora", split="train")
        df = ds.to_pandas()

        if "questions" in df.columns:
            df["text1"] = df["questions"].apply(
                lambda q: q[0]["text"] if isinstance(q, list) and len(q) > 0 else ""
            )
            df["text2"] = df["questions"].apply(
                lambda q: q[1]["text"] if isinstance(q, list) and len(q) > 1 else ""
            )
            df = df.drop(columns=["questions"])
        elif "question1" in df.columns and "question2" in df.columns:
            df = df.rename(columns={"question1": "text1", "question2": "text2"})

        if "is_duplicate" in df.columns:
            df["is_duplicate"] = df["is_duplicate"].astype(int)

        df = df[(df["text1"].str.len() > 0) & (df["text2"].str.len() > 0)].reset_index(drop=True)

        df_sample = df.sample(n=min(10000, len(df)), random_state=42)
        out_path = RAW_DIR / "quora_duplicates.csv"
        df_sample.to_csv(out_path, index=False)

        dup_count = df_sample["is_duplicate"].sum() if "is_duplicate" in df_sample.columns else "?"
        print(f"  Saved: {len(df_sample)} pairs → {out_path}")
        print(f"  Columns: {list(df_sample.columns)}")
        print(f"  Duplicates: {dup_count}/{len(df_sample)} ({dup_count/len(df_sample)*100:.1f}%)" if isinstance(dup_count, int) else "")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        print("  Manual: pip install datasets && python -c 'from datasets import load_dataset; ds = load_dataset(\"quora\")'")
        return False


# filler tickets when Drive export isn't there yet — still lets you run the pipeline
def generate_synthetic_tickets(n=500):
    print(f"\n[Synthetic] Generating {n} synthetic IT tickets...")
    import random

    categories = {
        "Hardware": {
            "components": ["Laptop", "Desktop", "Monitor", "Printer", "Keyboard", "Mouse", "Docking Station"],
            "issues": [
                "not turning on", "overheating", "screen flickering", "blue screen of death",
                "slow performance", "no display output", "battery not charging", "keyboard keys stuck",
                "USB ports not working", "fan making loud noise", "cracked screen",
                "not connecting to docking station", "freezing randomly"
            ],
            "resolutions": [
                "Replaced faulty component", "Updated BIOS firmware", "Performed hardware diagnostic",
                "Ordered replacement unit", "Cleaned internal fans", "Replaced battery",
                "Reseated RAM modules", "Updated device drivers"
            ]
        },
        "Software": {
            "components": ["Windows", "Office 365", "Outlook", "Teams", "VPN", "Antivirus", "SAP", "Salesforce"],
            "issues": [
                "crashing on startup", "not responding", "update failed", "license expired",
                "cannot login", "slow loading", "error code 0x80070005", "compatibility issue",
                "not syncing properly", "missing features after update", "corrupted installation",
                "memory leak causing slowdown", "plugin not working"
            ],
            "resolutions": [
                "Reinstalled application", "Cleared cache and temp files", "Applied latest patch",
                "Renewed license through admin portal", "Reset user profile",
                "Rolled back to previous version", "Configured firewall exceptions"
            ]
        },
        "Network": {
            "components": ["WiFi", "Ethernet", "VPN", "DNS", "Proxy", "Firewall"],
            "issues": [
                "cannot connect to WiFi", "intermittent disconnections", "slow internet speed",
                "VPN connection drops", "cannot access internal resources", "DNS resolution failure",
                "proxy authentication error", "IP conflict", "packet loss",
                "cannot reach specific websites", "network drive not mapping"
            ],
            "resolutions": [
                "Reset network adapter", "Flushed DNS cache", "Updated VPN client",
                "Assigned static IP", "Reconfigured proxy settings", "Replaced network cable",
                "Updated WiFi driver", "Added firewall rule exception"
            ]
        },
        "Account": {
            "components": ["Active Directory", "SSO", "Email", "MFA", "Password"],
            "issues": [
                "account locked out", "password expired", "MFA not working",
                "cannot access shared mailbox", "SSO login loop", "permission denied",
                "new hire needs account setup", "account deactivation request",
                "group membership change needed", "email forwarding setup"
            ],
            "resolutions": [
                "Unlocked account in AD", "Reset password and synced", "Re-enrolled MFA device",
                "Granted appropriate permissions", "Fixed SSO federation config",
                "Created new account with standard template", "Updated group memberships"
            ]
        },
        "Incident": {
            "components": ["Email Server", "Database", "Web App", "API Gateway", "Load Balancer", "Storage"],
            "issues": [
                "service outage affecting all users", "degraded performance across region",
                "data sync failure between systems", "SSL certificate expired",
                "disk space critical on production server", "memory exhaustion on app server",
                "scheduled maintenance overran window", "failover did not trigger",
                "backup job failed overnight", "unauthorized access attempt detected"
            ],
            "resolutions": [
                "Restarted affected services", "Scaled up server resources",
                "Renewed SSL certificate and redeployed", "Freed disk space and set alerts",
                "Performed manual failover", "Re-ran backup with increased timeout",
                "Blocked IP and reviewed access logs", "Applied emergency patch"
            ]
        }
    }

    severities = ["Critical", "High", "Medium", "Low"]
    severity_weights = [0.05, 0.15, 0.50, 0.30]
    statuses = ["Open", "In Progress", "Resolved", "Closed"]
    teams = ["IT Operations", "Network Engineering", "Help Desk", "Security", "DevOps"]
    environments = ["Windows 11", "Windows 10", "macOS 14", "Ubuntu 22.04", "Chrome OS"]

    tickets = []
    duplicate_pairs = []

    for i in range(n):
        category = random.choice(list(categories.keys()))
        cat_data = categories[category]
        component = random.choice(cat_data["components"])
        issue = random.choice(cat_data["issues"])
        severity = random.choices(severities, weights=severity_weights, k=1)[0]
        status = random.choice(statuses)
        team = random.choice(teams)
        env = random.choice(environments)

        templates = [
            f"My {component} is {issue}. I've been experiencing this since this morning. Environment: {env}.",
            f"Issue with {component}: {issue}. This is affecting my work. Running on {env}.",
            f"Hi, I need help. {component} keeps {issue}. Started yesterday. I'm on {env}.",
            f"Urgent: {component} - {issue}. Multiple users reporting similar issues. Environment is {env}.",
            f"The {component} has been {issue} for the past few hours. OS: {env}. Please help.",
        ]
        description = random.choice(templates)

        title_templates = [
            f"{component} {issue}",
            f"[{category}] {component} - {issue}",
            f"{component} problem: {issue}",
        ]
        title = random.choice(title_templates)

        resolution = random.choice(cat_data["resolutions"]) if status in ["Resolved", "Closed"] else ""
        resolution_notes = f"{resolution}. Verified with user that issue is fixed." if resolution else ""

        ticket = {
            "ticket_id": f"ITS-{i+1:05d}",
            "title": title,
            "description": description,
            "category": category,
            "component": component,
            "issue_type": "Incident" if category == "Incident" else random.choice(["Bug", "Service Request", "Question"]),
            "severity": severity,
            "priority": severity,
            "status": status,
            "assigned_team": team,
            "environment": env,
            "resolution": resolution_notes,
            "created_date": f"2025-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "steps_to_reproduce": f"1. Open {component}\n2. Attempt normal operation\n3. Observe: {issue}",
        }
        tickets.append(ticket)

        if random.random() < 0.15 and i > 0:
            orig_idx = random.randint(0, len(tickets) - 2)
            orig = tickets[orig_idx]

            dup_templates = [
                f"Same problem as others - {orig['component']} is {issue}. When will this be fixed?",
                f"I'm also having trouble with {orig['component']}. It's been {issue} all day on {env}.",
                f"Is anyone else seeing {orig['component']} {issue}? Started after the last update.",
            ]
            dup_desc = random.choice(dup_templates)

            duplicate_pairs.append({
                "ticket_1": orig["ticket_id"],
                "ticket_2": ticket["ticket_id"],
                "is_duplicate": 1,
                "similarity_type": "semantic"
            })

    df_tickets = pd.DataFrame(tickets)
    out_path = PROCESSED_DIR / "synthetic_tickets.csv"
    df_tickets.to_csv(out_path, index=False)
    print(f"  Saved: {len(df_tickets)} tickets → {out_path}")

    df_dups = pd.DataFrame(duplicate_pairs)
    out_path_dups = PROCESSED_DIR / "synthetic_duplicate_pairs.csv"
    df_dups.to_csv(out_path_dups, index=False)
    print(f"  Saved: {len(df_dups)} duplicate pairs → {out_path_dups}")

    print(f"\n  === Dataset Stats ===")
    print(f"  Total tickets: {len(df_tickets)}")
    print(f"  Categories: {df_tickets['category'].value_counts().to_dict()}")
    print(f"  Severities: {df_tickets['severity'].value_counts().to_dict()}")
    print(f"  Statuses: {df_tickets['status'].value_counts().to_dict()}")
    print(f"  Duplicate pairs: {len(df_dups)}")

    return df_tickets, df_dups


# minimal KB markdown so RAG has something to retrieve on day one
def generate_knowledge_base():
    print("\n[KB] Generating Knowledge Base documents...")

    kb_dir = Path("./data/knowledge_base")
    kb_dir.mkdir(parents=True, exist_ok=True)

    documents = {
        "runbook_vpn_troubleshooting.md": """# VPN Troubleshooting Runbook

## Symptom: VPN Connection Drops Intermittently

### Step 1: Check Client Version
Ensure the VPN client is updated to the latest version (v4.2+).
- Windows: Check via Settings > Apps > VPN Client
- macOS: Check via Applications > VPN Client > About

### Step 2: Network Diagnostics
1. Run `ping vpn-gateway.company.com` to verify connectivity
2. Check for packet loss: `tracert vpn-gateway.company.com`
3. Verify DNS resolution: `nslookup vpn-gateway.company.com`

### Step 3: Common Fixes
- **Split tunneling conflict**: Disable split tunneling in VPN settings
- **MTU mismatch**: Set MTU to 1400: `netsh interface ipv4 set subinterface "VPN" mtu=1400`
- **Firewall interference**: Add VPN client to firewall exceptions
- **Proxy conflict**: Disable proxy while connected to VPN

### Step 4: Escalation
If above steps don't resolve, collect:
- VPN client logs (export from client settings)
- Network trace (30 seconds during disconnect event)
- Escalate to Network Engineering team (Tier 2)

**Resolution time target: 30 minutes for Tier 1, 2 hours for Tier 2**
""",

        "runbook_password_reset.md": """# Password Reset & Account Lockout Runbook

## Symptom: Account Locked Out

### Step 1: Verify Identity
Confirm user identity through:
- Employee ID verification
- Manager confirmation (if remote)
- Security questions (if configured)

### Step 2: Check Lockout Reason
1. Open Active Directory Users and Computers
2. Find user account → Properties → Account tab
3. Check "Account is locked out" checkbox
4. Review Event Viewer on domain controller for Event ID 4740

### Common Causes:
- **Multiple failed login attempts**: Usually 5+ failures within 30 minutes
- **Cached credentials**: Old password stored on mobile device or mapped drive
- **Service account**: Application using expired credentials
- **Brute force attempt**: Check source IP in security logs

### Step 3: Unlock Account
1. In AD: Right-click user → Unlock Account
2. If password expired: Reset password, require change at next login
3. If MFA issue: Reset MFA enrollment in Azure AD / Okta admin portal

### Step 4: Prevent Recurrence
- Have user update credentials on all devices
- Check for service accounts using their credentials
- Enable self-service password reset if not already active

**Resolution time target: 15 minutes**
""",

        "runbook_email_issues.md": """# Email / Outlook Troubleshooting Runbook

## Symptom: Outlook Not Syncing / Cannot Send or Receive

### Step 1: Quick Checks
1. Verify internet connectivity
2. Check Microsoft 365 service health: https://status.office365.com
3. Try Outlook Web App (OWA) - if OWA works, issue is client-side

### Step 2: Client-Side Fixes
- **Clear Outlook cache**: Close Outlook → Delete files in `%localappdata%\\Microsoft\\Outlook\\RoamCache\\`
- **Repair Office**: Settings → Apps → Microsoft 365 → Modify → Quick Repair
- **Recreate profile**: Control Panel → Mail → Show Profiles → Add new profile
- **Disable add-ins**: File → Options → Add-ins → Manage COM Add-ins → Uncheck all

### Step 3: Server-Side Checks
- Check mailbox size (quota: 50GB for standard, 100GB for executives)
- Verify mail flow rules aren't blocking
- Check transport queue for stuck messages
- Review message trace in Exchange admin center

### Step 4: Shared Mailbox Issues
- Verify user has Full Access and Send As permissions
- Remove and re-add the shared mailbox in Outlook
- Check auto-mapping is enabled in Exchange

**Resolution time target: 20 minutes for client issues, 1 hour for server issues**
""",

        "runbook_hardware_laptop.md": """# Laptop Hardware Troubleshooting Runbook

## Symptom: Laptop Not Turning On / Blue Screen / Overheating

### Not Turning On
1. Perform hard reset: Remove power, hold power button 15 seconds
2. Try with only AC power (remove battery if removable)
3. Check power adapter LED indicator
4. Try external monitor to rule out display failure
5. If no POST: Likely motherboard or RAM failure → Escalate to hardware depot

### Blue Screen of Death (BSOD)
1. Note the stop code (e.g., IRQL_NOT_LESS_OR_EQUAL, PAGE_FAULT_IN_NONPAGED_AREA)
2. Check Event Viewer → Windows Logs → System for critical errors
3. Common fixes by stop code:
   - **IRQL_NOT_LESS_OR_EQUAL**: Driver issue → Update/rollback recent drivers
   - **CRITICAL_PROCESS_DIED**: System file corruption → Run `sfc /scannow`
   - **MEMORY_MANAGEMENT**: RAM issue → Run Windows Memory Diagnostic
   - **KERNEL_DATA_INPAGE_ERROR**: Disk issue → Run `chkdsk /f /r`
4. If recurring: Check for recent Windows updates, driver changes

### Overheating
1. Check Task Manager for high CPU processes
2. Verify fans are running (listen/feel for airflow)
3. Clean vents with compressed air
4. Check thermal paste age (if >3 years, may need replacement)
5. Use cooling pad as temporary measure
6. BIOS update may improve thermal management

**Escalation**: Hardware depot for physical repairs. Ship via standard IT logistics process.
""",

        "runbook_software_installation.md": """# Software Installation & License Troubleshooting

## Standard Software Catalog
All approved software is available through the Company Software Center.

### Installation Failures
1. **Error: Insufficient permissions**: User needs local admin → Submit elevation request
2. **Error: Disk space**: Need minimum 2GB free → Help user clear temp files
3. **Error: Compatibility**: Check system requirements against user's hardware
4. **Error: Network timeout**: Switch to wired connection, retry during off-peak

### License Issues
- **Office 365**: Licenses managed through Azure AD groups. Add user to correct license group.
- **Adobe Creative Cloud**: Limited seats. Check with department admin for availability.
- **Specialized software (SAP, Salesforce)**: Requires manager approval + specific role assignment.

### Common License Errors
- **"Product activation failed"**: Run `ospp.vbs /act` from admin command prompt
- **"License limit reached"**: Deactivate on old device first, or contact vendor
- **"Subscription expired"**: Verify user's license assignment in admin portal

**Resolution time target: 30 minutes for standard software, 24 hours for specialized**
""",

        "kb_article_teams_performance.md": """# Knowledge Base: Microsoft Teams Performance Issues

## Problem
Users report Teams running slowly, consuming high memory, or causing system lag.

## Root Cause
Teams is an Electron-based application that can consume significant RAM (1-2GB+).
Common triggers: many open chats, large files in channels, background activity.

## Solution
1. **Clear Teams cache** (most effective):
   - Close Teams completely (check system tray)
   - Delete contents of: `%appdata%\\Microsoft\\Teams\\Cache`
   - Also clear: `blob_storage`, `databases`, `GPUCache`, `IndexedDB`, `Local Storage`, `tmp`
   - Restart Teams

2. **Reduce memory usage**:
   - Close unused chats and channels
   - Disable GPU hardware acceleration: Settings → General → Uncheck "Disable GPU hardware acceleration"
   - Reduce notification frequency
   - Limit Teams startup: Settings → General → Uncheck "Auto-start Teams"

3. **New Teams app** (recommended):
   - Upgrade to "New Teams" (Microsoft Teams 2.0) which uses 50% less memory
   - Available through Settings → "Try the new Teams" toggle

## Prevention
- Regularly clear cache (monthly)
- Keep Teams updated to latest version
- Report persistent issues to vendor management for tracking
""",

        "kb_article_printer_setup.md": """# Knowledge Base: Network Printer Setup and Common Issues

## Setup: Adding a Network Printer
1. Open Settings → Bluetooth & Devices → Printers & Scanners
2. Click "Add device" → Select printer from list
3. If not found: Click "Add manually" → Enter IP address (see printer label)
4. Install driver from Company Software Center if prompted

## Common Issues

### Print Jobs Stuck in Queue
1. Open Services (services.msc) → Stop "Print Spooler"
2. Delete files in `C:\\Windows\\System32\\spool\\PRINTERS\\`
3. Restart Print Spooler service
4. Retry print job

### Printer Offline
1. Check physical connection (network cable, power)
2. Ping printer IP address
3. Remove and re-add printer
4. Check if print server is running (IT Operations can verify)

### Poor Print Quality
1. Run printer self-cleaning cycle (from printer menu)
2. Check toner/ink levels
3. Verify correct paper type is selected in print settings
4. If persistent: Submit maintenance request to Facilities

## Secure Print
All printers support secure print. Documents are held until user authenticates at the printer with their badge. Enable via: Print → Properties → Secure Print → Enter PIN.
"""
    }

    for filename, content in documents.items():
        filepath = kb_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Created: {filepath}")

    print(f"  Total KB documents: {len(documents)}")
    return list(documents.keys())


if __name__ == "__main__":
    _configure_stdout_utf8()

    parser = argparse.ArgumentParser(
        description="Step 1 — local-first: register your CSV, optional synthetic + KB. "
        "Public datasets only if you pass --with-public-datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=["major", "gitbugs", "jira", "github", "quora"],
        help="One optional public mirror (only that action runs, unless combined below).",
    )
    parser.add_argument(
        "--register-major-data",
        action="store_true",
        help="Only register the major ITS CSV (manifest for 02). No downloads.",
    )
    parser.add_argument(
        "--with-public-datasets",
        action="store_true",
        help="After other steps, try GitBugs / Jira helpers / GitHub script / Quora (network). "
        "Needs: pip install datasets requests (and HF access if hubs block you).",
    )
    parser.add_argument(
        "--major-path",
        default=str(default_major_its_csv_path()),
        help="Major tickets CSV path (default: its_ticket80.csv if present else its_tickets_80k.csv).",
    )
    parser.add_argument("--generate-synthetic", action="store_true", help="Only generate synthetic tickets")
    parser.add_argument("--generate-kb", action="store_true", help="Only write stub KB markdown")
    parser.add_argument("--synthetic-count", type=int, default=500, help="Synthetic ticket count")

    args = parser.parse_args()

    major_path = Path(args.major_path)
    explicit = any(
        [
            args.register_major_data,
            args.dataset,
            args.generate_synthetic,
            args.generate_kb,
            args.with_public_datasets,
        ]
    )

    if args.register_major_data:
        register_major_its_dataset(major_path)

    if args.dataset:
        {
            "major": lambda: register_major_its_dataset(major_path),
            "gitbugs": download_gitbugs,
            "jira": download_jira,
            "github": download_github,
            "quora": download_quora,
        }[args.dataset]()

    if args.generate_synthetic:
        generate_synthetic_tickets(args.synthetic_count)

    if args.generate_kb:
        generate_knowledge_base()

    if args.with_public_datasets:
        print("\n[optional] pulling public mirrors — skip if you only care about the 80k export")
        download_gitbugs()
        download_jira()
        download_github()
        download_quora()

    if not explicit:
        print("=" * 60)
        print("Local prep (no HuggingFace/GitHub unless you add --with-public-datasets)")
        print("=" * 60)
        print("  Tip: capstone artifacts live in the shared Google Drive folder — see README.md.")
        register_major_its_dataset(major_path)
        generate_synthetic_tickets(args.synthetic_count)
        generate_knowledge_base()

    print("\n✅ Data preparation complete!")
    print("Next: python 02_preprocess_data.py")

"""One-off: download Wikipedia plain extracts into data/knowledge_base/*.md (CC BY-SA)."""
import json
import pathlib
import re
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context
API = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&format=json&titles="


def fetch(wiki_title: str) -> tuple[str, str]:
    url = API + wiki_title.replace(" ", "%20")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "CapstoneITS-Michigan/1.0 (python; Wikipedia API; contact: course project)"
        },
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    p = list(d["query"]["pages"].values())[0]
    if "missing" in p:
        raise SystemExit(f"Missing page: {wiki_title}")
    return p["title"], p.get("extract") or ""


def wiki_to_md(extract: str) -> str:
    t = extract.replace("__NOTOC__", "")
    t = re.sub(r"^= (.+) =\s*\n", r"# \1\n\n", t, flags=re.M)
    t = re.sub(r"^== (.+) ==\s*\n", r"## \1\n\n", t, flags=re.M)
    t = re.sub(r"^=== (.+) ===\s*\n", r"### \1\n\n", t, flags=re.M)
    t = re.sub(r"^==== (.+) ====\s*\n", r"#### \1\n\n", t, flags=re.M)
    return t


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    outdir = root / "data" / "knowledge_base"
    outdir.mkdir(parents=True, exist_ok=True)
    articles = [
        ("Multi-factor_authentication", "kb_mfa_wikipedia"),
        ("Phishing", "kb_phishing_wikipedia"),
        ("Mobile_device_management", "kb_mdm_wikipedia"),
    ]
    for wiki_title, fname in articles:
        title, extract = fetch(wiki_title)
        head = f"""# {title}

> **Source:** [English Wikipedia](https://en.wikipedia.org/wiki/{wiki_title}) (plain-text extract).  
> **License:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).  
> **Retrieved:** 2026-04-26 (MediaWiki API). Keep this notice in derivative works (BY-SA).

---

"""
        path = outdir / f"{fname}.md"
        path.write_text(head + wiki_to_md(extract), encoding="utf-8")
        print(f"Wrote {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

"""Crude keyword router — maps noisy ticket text into one of four departments (demo / eval)."""
from __future__ import annotations

import re
from typing import Iterable


DEPARTMENTS: list[str] = [
    "IT Infrastructure & Platform",
    "End-User & Desktop Support",
    "Network & Security",
    "Applications & Data Services",
]


_DEPARTMENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Network & Security": (
        "vpn",
        "firewall",
        "network",
        "wifi",
        "wi-fi",
        "wireless",
        "dns",
        "dhcp",
        "proxy",
        "load balancer",
        "ssl",
        "certificate",
        "security",
        "mfa",
        "sso",
        "zero trust",
        "captive portal",
    ),
    "End-User & Desktop Support": (
        "laptop",
        "desktop",
        "monitor",
        "keyboard",
        "mouse",
        "printer",
        "print",
        "scanner",
        "dock",
        "docking",
        "windows",
        "macbook",
        "device",
        "hardware",
        "password",
        "account locked",
        "reset password",
        "outlook profile",
    ),
    "Applications & Data Services": (
        "application",
        "app",
        "software",
        "teams",
        "outlook",
        "email",
        "database",
        "sql",
        "api",
        "web app",
        "sap",
        "salesforce",
        "data",
        "report",
        "dashboard",
        "bug",
        "feature",
    ),
    "IT Infrastructure & Platform": (
        "server",
        "storage",
        "active directory",
        "directory",
        "vm",
        "virtual machine",
        "cloud",
        "aws",
        "azure",
        "backup",
        "domain controller",
        "platform",
        "infrastructure",
        "linux",
        "kubernetes",
        "container",
    ),
}


_DIRECT_ALIASES: dict[str, str] = {
    "hardware": "End-User & Desktop Support",
    "account": "End-User & Desktop Support",
    "network": "Network & Security",
    "security": "Network & Security",
    "software": "Applications & Data Services",
    "application": "Applications & Data Services",
    "applications": "Applications & Data Services",
    "data": "Applications & Data Services",
    "infrastructure": "IT Infrastructure & Platform",
    "platform": "IT Infrastructure & Platform",
}


def normalize_label(value: object) -> str:
    """Normalize source labels without losing meaningful words."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[_/\\|,;:(){}\[\]-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def map_to_department(*values: object, default: str = "IT Infrastructure & Platform") -> str:
    """Map raw ticket labels/text to one of the four final departments."""
    joined = " ".join(normalize_label(v) for v in values if str(v or "").strip())
    if not joined:
        return default

    for raw, dept in _DIRECT_ALIASES.items():
        if raw == joined or raw in joined:
            return dept

    scores = {dept: 0 for dept in DEPARTMENTS}
    for dept, keywords in _DEPARTMENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in joined:
                scores[dept] += 1

    best = max(scores.items(), key=lambda item: item[1])
    return best[0] if best[1] > 0 else default


def department_scores(values: Iterable[object]) -> dict[str, int]:
    """Return keyword hit counts for explainable data-readiness reports."""
    joined = " ".join(normalize_label(v) for v in values if str(v or "").strip())
    scores = {dept: 0 for dept in DEPARTMENTS}
    for dept, keywords in _DEPARTMENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in joined:
                scores[dept] += 1
    return scores


def classify_department(*values: object) -> dict[str, object]:
    """Classify text into the 4-department taxonomy with a lightweight confidence.

    This is intentionally cheap and deterministic. It is used before any LLM call
    so obvious cases route in milliseconds and provide explainable labels for
    future Model 2 training data.
    """
    joined = " ".join(normalize_label(v) for v in values if str(v or "").strip())
    if not joined:
        return {
            "department": "IT Infrastructure & Platform",
            "confidence": 0.25,
            "reason": "No routing text was available; using the default infrastructure queue.",
            "scores": {dept: 0 for dept in DEPARTMENTS},
            "source": "rules",
        }

    for raw, dept in _DIRECT_ALIASES.items():
        if raw == joined or raw in joined:
            scores = department_scores([joined])
            scores[dept] = max(scores.get(dept, 0), 2)
            return {
                "department": dept,
                "confidence": 0.86,
                "reason": f"Matched routing label or keyword '{raw}'.",
                "scores": scores,
                "source": "rules",
            }

    scores = department_scores([joined])
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_dept, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    if best_score <= 0:
        return {
            "department": "IT Infrastructure & Platform",
            "confidence": 0.35,
            "reason": "No strong department keywords matched; using the default infrastructure queue.",
            "scores": scores,
            "source": "rules",
        }

    margin = best_score - second_score
    confidence = min(0.92, 0.55 + (0.12 * best_score) + (0.08 * margin))
    return {
        "department": best_dept,
        "confidence": round(confidence, 3),
        "reason": f"Keyword evidence favored {best_dept} ({best_score} hits, margin {margin}).",
        "scores": scores,
        "source": "rules",
    }

"""
05 — wire step 04 to Ollama and get a "Resolution Blueprint" you can show a human.

This is the CLI twin of the FastAPI app; same retriever, same vibe, no browser.

Needs Ollama up (we used qwen3:8b for comfy typing, qwen3:4b for speed).

Usage:
  python 05_rag_pipeline.py
  python 05_rag_pipeline.py --query "VPN keeps dropping"
"""

import json
import argparse
import re
import requests
from typing import Any, Dict, List, Optional
from pathlib import Path
import importlib.util

import pandas as pd

# Import retrievers from 04_hybrid_retrieval.py (filename starts with a digit)
_retriever_path = Path(__file__).parent / "04_hybrid_retrieval.py"
_retriever_spec = importlib.util.spec_from_file_location("hybrid_retrieval_engine", _retriever_path)
if _retriever_spec is None or _retriever_spec.loader is None:
    raise ImportError(f"Cannot load retriever module from {_retriever_path}")
_retriever_module = importlib.util.module_from_spec(_retriever_spec)
_retriever_spec.loader.exec_module(_retriever_module)

HybridRetriever = _retriever_module.HybridRetriever
KBRetriever = _retriever_module.KBRetriever
CombinedRetriever = _retriever_module.CombinedRetriever

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3:8b"
MIN_RERANK_SCORE = -2.0

TICKET_COLLECTION = "its_tickets"
KB_COLLECTION = "its_knowledge_base"


# ──────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────
def strip_thinking_tokens(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def safe_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def is_relevant(doc: Dict[str, Any]) -> bool:
    score = doc.get("rerank_score", None)
    return score is None or float(score) >= MIN_RERANK_SCORE


def extract_json_object(text: str) -> Dict[str, Any] | None:
    cleaned = strip_thinking_tokens(text)
    decoder = json.JSONDecoder()
    for start, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


# ──────────────────────────────────────────────
# LLM Client (Ollama)
# ──────────────────────────────────────────────
class OllamaLLM:
    """Simple Ollama LLM client."""

    def __init__(self, model: str = MODEL_NAME, base_url: str = "http://localhost:11434"):
        self.model = model
        self.generate_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"

    def generate(self, prompt: str, temperature: float = 0.1,
                 max_tokens: int = 2048, system: str = "") -> str:
        """Generate text from a prompt without exposing thinking tokens."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        try:
            resp = requests.post(self.generate_url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return strip_thinking_tokens(data.get("response", ""))
        except requests.exceptions.ConnectionError:
            return "[ERROR] Cannot connect to Ollama. Make sure it's running: `ollama serve`"
        except Exception as e:
            return f"[ERROR] LLM generation failed: {e}"

    def chat(self, messages: List[Dict], temperature: float = 0.1,
             max_tokens: int = 1024, json_mode: bool = False) -> str:
        """Chat-style generation without returning model reasoning."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        try:
            resp = requests.post(self.chat_url, json=payload, timeout=120)
            resp.raise_for_status()
            return strip_thinking_tokens(resp.json().get("message", {}).get("content", ""))
        except requests.exceptions.ConnectionError:
            return "[ERROR] Cannot connect to Ollama. Make sure it's running: `ollama serve`"
        except Exception as e:
            return f"[ERROR] Chat failed: {e}"


# ──────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────
RESOLUTION_BLUEPRINT_JSON_PROMPT = """You are an IT helpdesk assistant writing a concise one-page resolution blueprint for ticket handlers.

Ticket:
{ticket_text}

Relevant historical tickets:
{similar_tickets}

Relevant knowledge base:
{kb_context}

Return ONLY valid JSON with this exact schema:
{{
  "recommended_actions": [
    "short action step",
    "short action step",
    "short action step"
  ],
  "similar_tickets": [
    {{
      "ticket_id": "ticket id",
      "component": "component",
      "status": "status",
      "summary": "one sentence describing the issue and the recorded resolution used; if no recorded resolution exists, say that explicitly"
    }}
  ],
  "escalate_if": [
    "condition for escalation and who should handle it"
  ]
}}

Rules:
- Maximum 4 recommended_actions.
- Maximum 3 similar_tickets.
- Maximum 2 escalate_if items.
- Use ONLY the retrieved evidence.
- Never reveal reasoning or chain-of-thought.
- Do not invent commands, resolutions, or policies.
- If evidence is weak, say the ticket match is partial or the KB is insufficient.
- JSON only.
"""

KB_QA_PROMPT = """You are a helpful IT knowledge base assistant. Answer the user's question using ONLY the provided context.

## User's Question
{query}

## Retrieved Knowledge Base Context
{context}

## Instructions
- Answer based ONLY on the provided context
- If the context doesn't contain the answer, say: "I don't have enough information in the knowledge base to answer this."
- Be concise and actionable
- Cite the source document when possible using [Source: document name]
"""

DUPLICATE_CHECK_PROMPT = """You are a ticket deduplication system. Determine if the new ticket is a duplicate of any existing tickets.

## New Ticket
{new_ticket}

## Potentially Similar Existing Tickets
{similar_tickets}

## Instructions
For each similar ticket, assess:
1. Is it describing the SAME issue? (yes/no/partial)
2. Confidence level (high/medium/low)
3. Brief reasoning

Then give a final verdict:
- DUPLICATE: Link to existing ticket [ID]
- RELATED: Similar but different issue, reference ticket [ID]
- UNIQUE: New issue, no duplicates found

Respond in JSON format:
{{
  "verdict": "DUPLICATE|RELATED|UNIQUE",
  "linked_ticket": "ticket_id or null",
  "confidence": "high|medium|low",
  "reasoning": "brief explanation",
  "assessments": [
    {{"ticket_id": "...", "is_duplicate": "yes|no|partial", "confidence": "...", "reason": "..."}}
  ]
}}
"""


# ──────────────────────────────────────────────
# RAG Pipeline
# ──────────────────────────────────────────────
class RAGPipeline:
    """
    Complete RAG pipeline for ITS:
    Ticket/Query → Hybrid Retrieval → Context Assembly → LLM Generation
    """

    def __init__(self):
        print("=" * 60)
        print("Initializing ITS RAG Pipeline")
        print("=" * 60)

        self.ticket_retriever = HybridRetriever(TICKET_COLLECTION)
        self.kb_retriever = HybridRetriever(KB_COLLECTION)
        self.ticket_lookup = self._load_ticket_lookup()

        print("\nInitializing LLM...")
        self.llm = OllamaLLM(model=MODEL_NAME)
        test = self.llm.generate("Say ready.", max_tokens=10)
        if "ERROR" in test:
            print(f"  ⚠ LLM not available: {test}")
            print(f"  Run: ollama pull {MODEL_NAME} && ollama serve")
        else:
            print(f"  ✅ LLM ready ({MODEL_NAME})")

        print("\n✅ RAG Pipeline initialized!\n")

    def _load_ticket_lookup(self) -> Dict[str, Dict[str, Any]]:
        lookup: Dict[str, Dict[str, Any]] = {}
        csv_path = Path(__file__).parent / "data" / "processed" / "all_tickets.csv"
        if not csv_path.exists():
            return lookup
        try:
            df = pd.read_csv(csv_path).fillna("")
            for _, row in df.iterrows():
                rec = row.to_dict()
                unified_id = str(rec.get("unified_id", "")).strip()
                ticket_id = str(rec.get("ticket_id", "")).strip()
                if unified_id:
                    lookup[unified_id] = rec
                if ticket_id and ticket_id not in lookup:
                    lookup[ticket_id] = rec
        except Exception:
            return {}
        return lookup

    def _ticket_record_for_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = str(doc.get("id", "")).strip()
        ticket_id = str(doc.get("metadata", {}).get("ticket_id", "")).strip()
        return self.ticket_lookup.get(doc_id) or self.ticket_lookup.get(ticket_id) or {}

    def _filter_results(self, results: List[Dict], limit: int) -> List[Dict]:
        seen = set()
        filtered = []
        for doc in results:
            doc_id = str(doc.get("id", ""))
            if doc_id in seen or not is_relevant(doc):
                continue
            seen.add(doc_id)
            filtered.append(doc)
            if len(filtered) >= limit:
                break
        return filtered

    def _format_ticket_results(self, results: List[Dict]) -> str:
        if not results:
            return "No sufficiently relevant historical tickets found."

        context_parts = []
        for i, doc in enumerate(results, start=1):
            meta = doc.get("metadata", {})
            rec = self._ticket_record_for_doc(doc)
            issue = safe_text(rec.get("title_clean") or meta.get("title") or rec.get("title") or "", 180)
            desc = safe_text(rec.get("description_clean") or rec.get("description") or doc.get("text", ""), 420)
            resolution = safe_text(rec.get("resolution_clean") or rec.get("resolution") or "", 220)
            context_parts.append(
                f"Ticket {i}: [{meta.get('ticket_id', rec.get('ticket_id', doc['id']))}]\n"
                f"Component: {meta.get('component', rec.get('component', 'General'))}\n"
                f"Status: {meta.get('status', rec.get('status', 'Unknown'))}\n"
                f"Issue: {issue}\n"
                f"Evidence: {desc}\n"
                f"Resolution Used: {resolution if resolution else 'No recorded resolution available in ticket history.'}\n"
            )
        return "\n".join(context_parts)

    def _format_kb_results(self, results: List[Dict]) -> str:
        if not results:
            return "No sufficiently relevant knowledge base articles found."

        context_parts = []
        for i, doc in enumerate(results, start=1):
            meta = doc.get("metadata", {})
            text = safe_text(doc.get("text", ""), 800)
            context_parts.append(
                f"KB {i}: [{meta.get('source_file', meta.get('doc_title', 'N/A'))}]\n"
                f"Section: {meta.get('section', 'N/A')}\n"
                f"Content: {text}\n"
            )
        return "\n".join(context_parts)

    def _normalize_blueprint(self, data: Dict[str, Any], similar_tickets: List[Dict]) -> Dict[str, Any]:
        actions = data.get("recommended_actions", [])
        if isinstance(actions, str):
            actions = [actions]
        actions = [safe_text(x, 220) for x in actions if str(x).strip()][:4]

        similar = data.get("similar_tickets", [])
        if isinstance(similar, str):
            similar = [similar]

        normalized_similar = []
        for item in similar[:3]:
            if isinstance(item, dict):
                normalized_similar.append({
                    "ticket_id": safe_text(item.get("ticket_id", ""), 50),
                    "component": safe_text(item.get("component", "General"), 60),
                    "status": safe_text(item.get("status", "Unknown"), 40),
                    "summary": safe_text(item.get("summary", ""), 260),
                })
            elif str(item).strip():
                normalized_similar.append({
                    "ticket_id": "",
                    "component": "General",
                    "status": "Unknown",
                    "summary": safe_text(item, 260),
                })

        if not normalized_similar:
            for doc in similar_tickets[:3]:
                meta = doc.get("metadata", {})
                rec = self._ticket_record_for_doc(doc)
                issue = safe_text(rec.get("title_clean") or meta.get("title") or rec.get("title") or doc.get("text", ""), 90)
                resolution = safe_text(rec.get("resolution_clean") or rec.get("resolution") or "No recorded resolution available.", 170)
                normalized_similar.append({
                    "ticket_id": safe_text(meta.get("ticket_id") or rec.get("ticket_id") or doc.get("id"), 50),
                    "component": safe_text(meta.get("component") or rec.get("component") or "General", 60),
                    "status": safe_text(meta.get("status") or rec.get("status") or "Unknown", 40),
                    "summary": safe_text(f"{issue}. Resolution used: {resolution}", 260),
                })

        escalate_if = data.get("escalate_if", [])
        if isinstance(escalate_if, str):
            escalate_if = [escalate_if]
        escalate_if = [safe_text(x, 220) for x in escalate_if if str(x).strip()][:2]

        if not actions:
            actions = ["No confident KB-backed action was found. Review the closest ticket history before making changes."]
        if not escalate_if:
            escalate_if = ["The symptom cannot be matched to the retrieved KB guidance or the issue persists after the documented steps."]

        return {
            "recommended_actions": actions,
            "similar_tickets": normalized_similar,
            "escalate_if": escalate_if,
        }

    def _render_blueprint(self, data: Dict[str, Any]) -> str:
        lines = ["## Resolution Blueprint", "", "**Recommended Actions**"]
        for idx, action in enumerate(data["recommended_actions"], start=1):
            lines.append(f"{idx}. {action}")

        lines.extend(["", "**From Similar Tickets**"])
        for item in data["similar_tickets"]:
            lines.append(
                f"- [{item.get('ticket_id') or 'N/A'}] ({item.get('component') or 'General'} · "
                f"{item.get('status') or 'Unknown'}): {item.get('summary') or 'No summary available.'}"
            )

        lines.extend(["", "**Escalate If**"])
        for item in data["escalate_if"]:
            lines.append(f"- {item}")
        return "\n".join(lines).strip()

    # ── Resolution Blueprint Generation ──
    def generate_resolution(self, ticket_text: str,
                            ticket_top_k: int = 3,
                            kb_top_k: int = 3,
                            verbose: bool = False) -> Dict:
        """
        Generate a Resolution Blueprint for a ticket.
        Retrieves similar tickets + KB articles → LLM generates structured output.
        """
        if verbose:
            print("  Retrieving similar tickets...")
        similar_tickets = self._filter_results(
            self.ticket_retriever.search(ticket_text, top_k=ticket_top_k),
            limit=ticket_top_k,
        )

        if verbose:
            print("  Retrieving KB articles...")
        kb_results = self._filter_results(
            self.kb_retriever.search(ticket_text, top_k=kb_top_k),
            limit=kb_top_k,
        )

        prompt = RESOLUTION_BLUEPRINT_JSON_PROMPT.format(
            ticket_text=ticket_text,
            similar_tickets=self._format_ticket_results(similar_tickets),
            kb_context=self._format_kb_results(kb_results),
        )

        if verbose:
            print("  Generating structured blueprint with LLM...")
        raw_response = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an IT helpdesk assistant. Never reveal reasoning or chain-of-thought. "
                        "Return JSON only. Use only retrieved evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=700,
            json_mode=True,
        )

        parsed = extract_json_object(raw_response)
        if parsed is None:
            parsed = {"recommended_actions": [self._format_kb_results(kb_results)[:220]]}
        response = self._render_blueprint(self._normalize_blueprint(parsed, similar_tickets))

        return {
            "resolution": response,
            "raw_resolution": raw_response,
            "similar_tickets": similar_tickets,
            "kb_articles": kb_results,
            "ticket_text": ticket_text,
        }

    # ── Knowledge Base Q&A ──
    def answer_question(self, query: str, top_k: int = 5,
                        verbose: bool = False) -> Dict:
        if verbose:
            print("  Retrieving from KB...")
        kb_results = self._filter_results(self.kb_retriever.search(query, top_k=top_k), limit=top_k)

        prompt = KB_QA_PROMPT.format(
            query=query,
            context=self._format_kb_results(kb_results),
        )

        if verbose:
            print("  Generating answer...")
        response = self.llm.generate(prompt, temperature=0.1)

        return {
            "answer": response,
            "sources": kb_results,
            "query": query,
        }

    # ── Duplicate Detection ──
    def check_duplicate(self, new_ticket_text: str,
                        top_k: int = 5,
                        verbose: bool = False) -> Dict:
        if verbose:
            print("  Searching for similar tickets...")
        similar = self._filter_results(self.ticket_retriever.search(new_ticket_text, top_k=top_k), limit=top_k)

        prompt = DUPLICATE_CHECK_PROMPT.format(
            new_ticket=new_ticket_text,
            similar_tickets=self._format_ticket_results(similar),
        )

        if verbose:
            print("  Analyzing with LLM...")
        response = self.llm.generate(prompt, temperature=0.0)

        try:
            verdict = extract_json_object(response) or {"raw_response": response}
        except Exception:
            verdict = {"raw_response": response}

        return {
            "verdict": verdict,
            "similar_tickets": similar,
            "new_ticket": new_ticket_text,
        }

    # ── Find Similar Tickets (no LLM) ──
    def find_similar(self, query: str, top_k: int = 5,
                     filters: Optional[Dict] = None) -> List[Dict]:
        return self._filter_results(
            self.ticket_retriever.search(query, top_k=top_k, filters=filters),
            limit=top_k,
        )


# ──────────────────────────────────────────────
# Pretty Printing
# ──────────────────────────────────────────────
def print_resolution(result: Dict):
    print("\n" + "═" * 60)
    print("  RESOLUTION BLUEPRINT")
    print("═" * 60)
    print(f"\n📋 Ticket: {result['ticket_text'][:100]}...")
    print(f"\n{'─' * 60}")
    print(result["resolution"])
    print(f"{'─' * 60}")
    print(f"\n📎 Based on {len(result['similar_tickets'])} similar tickets "
          f"and {len(result['kb_articles'])} KB articles")


def print_answer(result: Dict):
    print("\n" + "═" * 60)
    print(f"  Q: {result['query']}")
    print("═" * 60)
    print(f"\n{result['answer']}")
    print(f"\n📎 Sources: {len(result['sources'])} KB articles retrieved")


def print_duplicate_check(result: Dict):
    print("\n" + "═" * 60)
    print("  DUPLICATE CHECK")
    print("═" * 60)
    verdict = result.get("verdict", {})
    if isinstance(verdict, dict) and "verdict" in verdict:
        print(f"\n  Verdict: {verdict['verdict']}")
        print(f"  Confidence: {verdict.get('confidence', 'N/A')}")
        print(f"  Reasoning: {verdict.get('reasoning', 'N/A')}")
        if verdict.get("linked_ticket"):
            print(f"  Linked to: {verdict['linked_ticket']}")
    else:
        print(f"\n{verdict.get('raw_response', verdict)}")


# ──────────────────────────────────────────────
# Interactive Mode
# ──────────────────────────────────────────────
def interactive_mode():
    rag = RAGPipeline()

    print("\n" + "=" * 60)
    print("ITS RAG Pipeline - Interactive Mode")
    print("=" * 60)
    print("Commands:")
    print("  /resolve <ticket text>   - Generate Resolution Blueprint")
    print("  /ask <question>          - KB Q&A")
    print("  /dedup <ticket text>     - Check for duplicates")
    print("  /similar <query>         - Find similar tickets")
    print("  /quit                    - Exit")
    print("=" * 60)

    while True:
        user_input = input("\n> ").strip()
        if not user_input:
            continue

        if user_input.lower() in ["/quit", "quit", "exit"]:
            break

        if user_input.startswith("/resolve "):
            ticket_text = user_input[9:]
            result = rag.generate_resolution(ticket_text, verbose=True)
            print_resolution(result)

        elif user_input.startswith("/ask "):
            question = user_input[5:]
            result = rag.answer_question(question, verbose=True)
            print_answer(result)

        elif user_input.startswith("/dedup "):
            ticket_text = user_input[7:]
            result = rag.check_duplicate(ticket_text, verbose=True)
            print_duplicate_check(result)

        elif user_input.startswith("/similar "):
            query = user_input[9:]
            results = rag.find_similar(query, top_k=5)
            for i, r in enumerate(results):
                meta = r.get("metadata", {})
                print(f"  [{i+1}] {meta.get('title', r['text'][:60])}")
                print(f"      Score: {r.get('rerank_score', r.get('score', 0)):.4f} | "
                      f"{meta.get('severity', '')} | {meta.get('category', '')}")

        else:
            result = rag.generate_resolution(user_input, verbose=True)
            print_resolution(result)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ITS RAG Pipeline")
    parser.add_argument("--query", type=str, help="Query text")
    parser.add_argument("--mode", choices=["resolution", "qa", "dedup", "similar"],
                        default="resolution", help="Pipeline mode")
    parser.add_argument("--ticket-text", type=str, help="Ticket text for resolution/dedup")
    args = parser.parse_args()

    if args.query or args.ticket_text:
        rag = RAGPipeline()
        text = args.ticket_text or args.query

        if args.mode == "resolution":
            result = rag.generate_resolution(text, verbose=True)
            print_resolution(result)
        elif args.mode == "qa":
            result = rag.answer_question(text, verbose=True)
            print_answer(result)
        elif args.mode == "dedup":
            result = rag.check_duplicate(text, verbose=True)
            print_duplicate_check(result)
        elif args.mode == "similar":
            results = rag.find_similar(text, top_k=5)
            for i, r in enumerate(results):
                meta = r.get("metadata", {})
                print(f"  [{i+1}] {meta.get('title', 'N/A')} (score: {r.get('score', 0):.4f})")
    else:
        interactive_mode()

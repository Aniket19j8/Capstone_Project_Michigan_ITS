"""
ITS RAG - Step 5: RAG Pipeline (Retrieval-Augmented Generation)
Connects the hybrid retriever to the LLM for grounded resolution generation.

Features:
  - Resolution Blueprint generation (Agent Assist)
  - Grounded answers with citations
  - Ticket similarity search with context
  - Knowledge base Q&A

Requires: Ollama running with qwen3:8b (or configured model)
  ollama pull qwen3:8b
  ollama serve

Usage:
  python 05_rag_pipeline.py                          # Interactive
  python 05_rag_pipeline.py --query "VPN keeps dropping"
  python 05_rag_pipeline.py --mode resolution --ticket-text "My VPN disconnects every 10 minutes"
"""

import json
import argparse
import requests
from typing import List, Dict, Optional
from pathlib import Path
import importlib.util

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
MODEL_NAME = "qwen3:8b"  # Change to llama3.1:8b, mistral:7b, etc.

TICKET_COLLECTION = "its_tickets"
KB_COLLECTION = "its_knowledge_base"


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
        """Generate text from a prompt."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        try:
            resp = requests.post(self.generate_url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            return "[ERROR] Cannot connect to Ollama. Make sure it's running: `ollama serve`"
        except Exception as e:
            return f"[ERROR] LLM generation failed: {e}"

    def chat(self, messages: List[Dict], temperature: float = 0.1) -> str:
        """Chat-style generation."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        }
        try:
            resp = requests.post(self.chat_url, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "").strip()
        except requests.exceptions.ConnectionError:
            return "[ERROR] Cannot connect to Ollama. Make sure it's running: `ollama serve`"
        except Exception as e:
            return f"[ERROR] Chat failed: {e}"


# ──────────────────────────────────────────────
# Prompt Templates
# ──────────────────────────────────────────────
RESOLUTION_BLUEPRINT_PROMPT = """You are an expert IT support agent. Based on the user's ticket and the retrieved context below, generate a Resolution Blueprint.

## User's Ticket
{ticket_text}

## Similar Historical Tickets (resolved)
{similar_tickets}

## Relevant Knowledge Base Articles
{kb_context}

## Instructions
Generate a Resolution Blueprint with these sections:
1. **Issue Summary**: One-line summary of the problem
2. **Root Cause Analysis**: Most likely root cause based on similar tickets
3. **Recommended Steps**: Step-by-step resolution guide (numbered)
4. **Cited Sources**: Which historical tickets or KB articles support each recommendation
5. **Escalation Path**: If these steps don't work, who to escalate to and what info to provide

IMPORTANT:
- Base your answer ONLY on the provided context. Do not hallucinate solutions.
- If the context doesn't contain enough information, say so explicitly.
- Cite sources using [Ticket: ID] or [KB: document name] format.
- Be specific and actionable. No vague advice.
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

        # Initialize retrievers
        self.ticket_retriever = HybridRetriever(TICKET_COLLECTION)
        self.kb_retriever = HybridRetriever(KB_COLLECTION)

        # Initialize LLM
        print("\nInitializing LLM...")
        self.llm = OllamaLLM(model=MODEL_NAME)
        test = self.llm.generate("Say 'ready' if you're working.", max_tokens=10)
        if "ERROR" in test:
            print(f"  ⚠ LLM not available: {test}")
            print(f"  Run: ollama pull {MODEL_NAME} && ollama serve")
        else:
            print(f"  ✅ LLM ready ({MODEL_NAME})")

        print("\n✅ RAG Pipeline initialized!\n")

    def _format_ticket_results(self, results: List[Dict]) -> str:
        """Format ticket search results as context string."""
        if not results:
            return "No similar tickets found."

        context_parts = []
        for i, doc in enumerate(results):
            meta = doc.get("metadata", {})
            text = doc.get("text", "")[:500]
            score = doc.get("rerank_score", doc.get("rrf_score", doc.get("score", 0)))
            context_parts.append(
                f"--- Ticket {i+1} [ID: {meta.get('ticket_id', doc['id'])}] "
                f"(relevance: {score:.3f}) ---\n"
                f"Title: {meta.get('title', 'N/A')}\n"
                f"Category: {meta.get('category', 'N/A')} | "
                f"Severity: {meta.get('severity', 'N/A')} | "
                f"Status: {meta.get('status', 'N/A')}\n"
                f"Content: {text}\n"
            )
        return "\n".join(context_parts)

    def _format_kb_results(self, results: List[Dict]) -> str:
        """Format KB search results as context string."""
        if not results:
            return "No relevant knowledge base articles found."

        context_parts = []
        for i, doc in enumerate(results):
            meta = doc.get("metadata", {})
            text = doc.get("text", "")[:800]
            context_parts.append(
                f"--- KB Article {i+1} [Source: {meta.get('source_file', meta.get('doc_title', 'N/A'))}] ---\n"
                f"Section: {meta.get('section', 'N/A')}\n"
                f"Content:\n{text}\n"
            )
        return "\n".join(context_parts)

    # ── Resolution Blueprint Generation ──
    def generate_resolution(self, ticket_text: str,
                            ticket_top_k: int = 3,
                            kb_top_k: int = 3,
                            verbose: bool = False) -> Dict:
        """
        Generate a Resolution Blueprint for a ticket.
        Retrieves similar tickets + KB articles → LLM generates resolution.
        """
        if verbose:
            print(f"  Retrieving similar tickets...")
        similar_tickets = self.ticket_retriever.search(ticket_text, top_k=ticket_top_k)

        if verbose:
            print(f"  Retrieving KB articles...")
        kb_results = self.kb_retriever.search(ticket_text, top_k=kb_top_k)

        # Assemble prompt
        prompt = RESOLUTION_BLUEPRINT_PROMPT.format(
            ticket_text=ticket_text,
            similar_tickets=self._format_ticket_results(similar_tickets),
            kb_context=self._format_kb_results(kb_results),
        )

        if verbose:
            print(f"  Generating resolution with LLM...")
        response = self.llm.generate(prompt, temperature=0.1)

        return {
            "resolution": response,
            "similar_tickets": similar_tickets,
            "kb_articles": kb_results,
            "ticket_text": ticket_text,
        }

    # ── Knowledge Base Q&A ──
    def answer_question(self, query: str, top_k: int = 5,
                        verbose: bool = False) -> Dict:
        """Answer a question using the knowledge base."""
        if verbose:
            print(f"  Retrieving from KB...")
        kb_results = self.kb_retriever.search(query, top_k=top_k)

        prompt = KB_QA_PROMPT.format(
            query=query,
            context=self._format_kb_results(kb_results),
        )

        if verbose:
            print(f"  Generating answer...")
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
        """Check if a ticket is a duplicate of existing tickets."""
        if verbose:
            print(f"  Searching for similar tickets...")
        similar = self.ticket_retriever.search(new_ticket_text, top_k=top_k)

        prompt = DUPLICATE_CHECK_PROMPT.format(
            new_ticket=new_ticket_text,
            similar_tickets=self._format_ticket_results(similar),
        )

        if verbose:
            print(f"  Analyzing with LLM...")
        response = self.llm.generate(prompt, temperature=0.0)

        # Try to parse JSON response
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                verdict = json.loads(json_match.group())
            else:
                verdict = {"raw_response": response}
        except:
            verdict = {"raw_response": response}

        return {
            "verdict": verdict,
            "similar_tickets": similar,
            "new_ticket": new_ticket_text,
        }

    # ── Find Similar Tickets (no LLM) ──
    def find_similar(self, query: str, top_k: int = 5,
                     filters: Optional[Dict] = None) -> List[Dict]:
        """Pure retrieval - find similar tickets without LLM."""
        return self.ticket_retriever.search(query, top_k=top_k, filters=filters)


# ──────────────────────────────────────────────
# Pretty Printing
# ──────────────────────────────────────────────
def print_resolution(result: Dict):
    """Pretty print a resolution blueprint."""
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
    """Pretty print a KB answer."""
    print("\n" + "═" * 60)
    print(f"  Q: {result['query']}")
    print("═" * 60)
    print(f"\n{result['answer']}")
    print(f"\n📎 Sources: {len(result['sources'])} KB articles retrieved")


def print_duplicate_check(result: Dict):
    """Pretty print duplicate check results."""
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
    """Interactive RAG pipeline."""
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
            # Default: treat as resolution request
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

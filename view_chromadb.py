import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import chromadb


DEFAULT_DB_PATH = Path("data/chroma_db")


def to_jsonable(obj: Any) -> Any:
    """Best-effort conversion for JSON printing."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return to_jsonable(obj.tolist())
    return obj


def get_client(db_path: Path):
    return chromadb.PersistentClient(path=str(db_path))


def list_collections(client) -> List[str]:
    collections = client.list_collections()
    names = []
    for c in collections:
        if hasattr(c, "name"):
            names.append(c.name)
        else:
            names.append(str(c))
    return names


def print_collections(client) -> None:
    names = list_collections(client)
    if not names:
        print("No collections found.")
        return
    print("Collections:")
    for name in names:
        col = client.get_collection(name)
        print(f"  - {name} (count={col.count()})")


def inspect_collection(client, name: str, peek: int) -> None:
    col = client.get_collection(name)
    print(f"Collection: {name}")
    print(f"Count: {col.count()}")

    if peek > 0:
        data = col.peek(limit=peek)
        print(f"\nSample documents (limit={peek}):")
        print(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False))


def query_collection(client, name: str, query_text: str, n_results: int) -> None:
    col = client.get_collection(name)
    result = col.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    print(f"Query results for collection '{name}':")
    print(json.dumps(to_jsonable(result), indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local ChromaDB.")
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="Path to ChromaDB persist directory (default: data/chroma_db)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection name to inspect (example: its_tickets)",
    )
    parser.add_argument(
        "--peek",
        type=int,
        default=5,
        help="How many sample records to show with --collection (default: 5)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional query text to search inside --collection",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=3,
        help="Top-k results for --query (default: 3)",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"ChromaDB path not found: {db_path}")

    client = get_client(db_path)

    if not args.collection:
        print_collections(client)
        return

    inspect_collection(client, args.collection, args.peek)
    if args.query:
        print()
        query_collection(client, args.collection, args.query, args.n_results)


if __name__ == "__main__":
    main()

import argparse
from lib.hybrid_search import (
    normalize_command,
    weighted_search_command,
)
from lib.search_utils import DEFAULT_SEARCH_LIMIT

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument("scores", type=float, nargs='+', help="scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for semantic search")
    weighted_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_command(args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            data = weighted_search_command(args.query, args.alpha, args.limit)
            for i, item in enumerate(data, 1):
                print(f"{i}. {item['metadata']['title']}")
                print(f"   Hybrid Score: {item['hybrid_score']:.4f}\n")
                print(f"   BM25 Score: {item['bm25_score']:.4f}, Semantic Score: {item['semantic_score']:.4f}\n")
                print(f"   {item['metadata']['description'][:100]}...\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
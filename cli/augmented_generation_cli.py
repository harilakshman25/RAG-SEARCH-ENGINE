import argparse
from lib.augmented_generation import (
    perform_rag,
    summarize,
    search_with_citations,
)

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Generate a summary based on provided context"
    )
    summarize_parser.add_argument("query", type=str, help="Gives a summary of the context retrieved based on the query")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Number of documents to retrieve for context")

    citations_parser = subparsers.add_parser(
        "citations", help="Generate citations for provided text"
    )
    citations_parser.add_argument("query", type=str, help="Text to generate citations for")
    citations_parser.add_argument("--limit", type=int, default=5, help="Number of citations to generate")

    args = parser.parse_args()

    match args.command:
        case "rag":
            perform_rag(args.query)
        case "summarize":
            summarize(args.query, args.limit)
        case "citations":
            search_with_citations(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
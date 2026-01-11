#!/usr/bin/env python3

import argparse
from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.semantic_search import (
    verify_model,
    embed_text,
    verify_embeddings,
    search_command,
    chunk_command,
    semantic_chunk_command,
    embed_chunks_command
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("verify", help="Verify the semantic search model")

    embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for text")
    embed_parser.add_argument("text", type=str)

    subparsers.add_parser("verify_embeddings", help="Verify stored embeddings")

    search_parser = subparsers.add_parser("search", help="Perform semantic search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT)

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text")
    chunk_parser.add_argument("text", type=str)
    chunk_parser.add_argument("--chunk_size", type=int, default=200)
    chunk_parser.add_argument("--overlap", type=int, default=5)

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantic Chunk text")
    semantic_chunk_parser.add_argument("text", type=str)
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4)
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0)

    subparsers.add_parser("embed_chunks", help="Generate chunked embeddings")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            chunks = chunk_command(args.text, args.chunk_size, args.overlap)
            for i, chunk in enumerate(chunks, 1):
                print(f"Chunk {i}:\n{chunk}\n")
        case "semantic_chunk":
            chunks = semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
            for i, chunk in enumerate(chunks, 1):
                print(f"Chunk {i}:\n{chunk}\n")
        case "embed_chunks":
            embed_chunks_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
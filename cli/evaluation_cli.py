import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies

def calculate_metrics(retrieved_titles: list[str], relevant_titles: list[str]):
    """
    Calculates Precision, Recall, and F1 Score.
    
    Args:
        retrieved_titles: List of titles returned by the search engine.
        relevant_titles: List of relevant titles from the golden dataset.
        
    Returns:
        tuple: (precision, recall, f1)
    """
    relevant_set = set(relevant_titles)
    retrieved_set = set(retrieved_titles)
    
    # Calculate intersection (relevant items found)
    relevant_retrieved = [t for t in retrieved_set if t in relevant_set]
    num_relevant_retrieved = len(relevant_retrieved)
    
    k = len(retrieved_titles)
    total_relevant = len(relevant_titles)
    
    # Precision@K: Portion of retrieved items that are relevant
    precision = num_relevant_retrieved / k if k > 0 else 0.0
    
    # Recall@K: Portion of relevant items that were retrieved
    recall = num_relevant_retrieved / total_relevant if total_relevant > 0 else 0.0
    
    # F1 Score: Harmonic mean of Precision and Recall
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, "data", "golden_dataset.json")

    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(base_dir, "..", "data", "golden_dataset.json")

    if not os.path.exists(dataset_path):
        print(f"Error: Could not find golden_dataset.json at {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        golden_data = json.load(f)

    documents = load_movies()
    search_engine = HybridSearch(documents)

    print(f"k={limit}\n")

    for case in golden_data["test_cases"]:
        query = case["query"]
        relevant_docs = case["relevant_docs"]
        
        results = search_engine.rrf_search(query, k=60, limit=limit)
        
        retrieved_titles = [item["metadata"]["title"] for item in results]
        
        precision, recall, f1 = calculate_metrics(retrieved_titles, relevant_docs)
        
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        
        retrieved_str = ", ".join(retrieved_titles)
        relevant_str = ", ".join(relevant_docs)
        
        print(f"  - Retrieved: {retrieved_str}")
        print(f"  - Relevant: {relevant_str}\n")

if __name__ == "__main__":
    main()

import json

from pathlib import Path

from app.retrieval.semantic_search import (
    search_chunks,
    hybrid_search,
)


DATASET_PATH = Path("evaluation/retrieval_cases.json")
K = 3


def calculate_metrics(
    results: list[dict],
    relevant_ids: set[str],
    k: int,
) -> tuple[float, float]:

    retrieved_ids = {
        result["chunk_id"]
        for result in results[:k]
    }

    correct = len(retrieved_ids & relevant_ids)

    precision = correct / k
    recall = correct / len(relevant_ids)

    return precision, recall


def evaluate() -> None:

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    cases = dataset["cases"]

    if not cases:
        raise ValueError("Değerlendirme soruları bulunamadı.")

    totals = {
        "Semantic": {"precision": 0.0, "recall": 0.0},
        "Hybrid": {"precision": 0.0, "recall": 0.0},
    }

    for case in cases:

        question = case["question"]
        relevant_ids = set(case["relevant_chunk_ids"])

        if not relevant_ids:
            raise ValueError("İlgili chunk listesi boş olamaz.")

        print(f"\nSoru: {question}")

        methods = {
            "Semantic": search_chunks(question, limit=K),
            "Hybrid": hybrid_search(question, limit=K),
        }

        for name, results in methods.items():

            precision, recall = calculate_metrics(
                results,
                relevant_ids,
                K,
            )

            totals[name]["precision"] += precision
            totals[name]["recall"] += recall

            print(f"\n{name} Search")
            print(f"Precision@{K}: %{precision * 100:.2f}")
            print(f"Recall@{K}: %{recall * 100:.2f}")

            for result in results:
                print(f"  {result['chunk_id']}")

    print("\n--- ORTALAMA SONUÇLAR ---")

    for name, scores in totals.items():

        avg_precision = scores["precision"] / len(cases)
        avg_recall = scores["recall"] / len(cases)

        print(f"\n{name} Search")
        print(f"Precision@{K}: %{avg_precision * 100:.2f}")
        print(f"Recall@{K}: %{avg_recall * 100:.2f}")


if __name__ == "__main__":
    evaluate()

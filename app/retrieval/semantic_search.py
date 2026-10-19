
import argparse
import json
import re
import uuid

from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "medical_chunks"

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DATABASE_PATH = Path("data/qdrant")


def load_chunks() -> list[dict]:

    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError("İşlenmiş PDF verileri bulunamadı.")

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not chunks:
        raise ValueError("Metin parçaları bulunamadı.")

    return chunks


def tokenize(text: str) -> list[str]:

    text = text.replace("İ", "i").replace("I", "ı").lower()

    return re.findall(r"\w+", text)


def index_chunks() -> int:

    chunks = load_chunks()

    embedding_model = TextEmbedding(model_name=MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    vectors = list(
        embedding_model.embed(texts, batch_size=4)
    )

    DATABASE_PATH.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(path=str(DATABASE_PATH))

    try:
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=len(vectors[0]),
                distance=models.Distance.COSINE,
            ),
        )

        points = []

        for chunk, vector in zip(chunks, vectors):

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    chunk["chunk_id"],
                )
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=chunk,
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

    finally:
        client.close()

    return len(points)


def search_chunks(question: str, limit: int = 3) -> list[dict]:

    if not question.strip():
        raise ValueError("Arama sorusu boş olamaz.")

    if limit <= 0:
        raise ValueError("Sonuç sayısı sıfırdan büyük olmalıdır.")

    if not DATABASE_PATH.exists():
        raise FileNotFoundError("Vektör veritabanı bulunamadı.")

    client = QdrantClient(path=str(DATABASE_PATH))

    try:
        if not client.collection_exists(COLLECTION_NAME):
            raise ValueError("Önce metin parçalarını indekslemelisiniz.")

        embedding_model = TextEmbedding(model_name=MODEL_NAME)

        question_vector = next(
            embedding_model.embed([question])
        )

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=question_vector.tolist(),
            limit=limit,
            with_payload=True,
        )

        return [
            {
                "chunk_id": point.payload["chunk_id"],
                "score": round(point.score, 4),
                "page_number": point.payload["page_number"],
                "source_filename": point.payload["source_filename"],
                "text": point.payload["text"],
            }
            for point in results.points
        ]

    finally:
        client.close()


def hybrid_search(question: str, limit: int = 3) -> list[dict]:

    if not question.strip():
        raise ValueError("Arama sorusu boş olamaz.")

    if limit <= 0:
        raise ValueError("Sonuç sayısı sıfırdan büyük olmalıdır.")

    chunks = load_chunks()

    candidate_limit = min(max(limit, 10), len(chunks))

    semantic_results = search_chunks(question, candidate_limit)

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    bm25_scores = bm25.get_scores(tokenize(question))

    ranked_bm25 = sorted(
        [
            (chunk, float(score))
            for chunk, score in zip(chunks, bm25_scores)
            if score > 0
        ],
        key=lambda item: item[1],
        reverse=True,
    )[:candidate_limit]

    fused = {}

    for rank, result in enumerate(semantic_results, start=1):

        chunk_id = result["chunk_id"]

        fused[chunk_id] = {
            **result,
            "rrf_score": 1 / (60 + rank),
            "semantic_rank": rank,
            "bm25_rank": None,
        }

    for rank, (chunk, _) in enumerate(ranked_bm25, start=1):

        chunk_id = chunk["chunk_id"]

        if chunk_id not in fused:
            fused[chunk_id] = {
                **chunk,
                "rrf_score": 0.0,
                "semantic_rank": None,
                "bm25_rank": None,
            }

        fused[chunk_id]["rrf_score"] += 1 / (60 + rank)
        fused[chunk_id]["bm25_rank"] = rank

    results = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return results[:limit]


def main() -> None:

    parser = argparse.ArgumentParser(
        description="MedIntel AI Arama Sistemi"
    )

    parser.add_argument(
        "action",
        choices=["index", "search", "hybrid"],
    )

    parser.add_argument("question", nargs="?")

    args = parser.parse_args()

    try:
        if args.action == "index":

            total = index_chunks()

            print(f"Başarılı! {total} metin parçası indekslendi.")

        else:
            if not args.question:
                raise ValueError("Arama sorusu girmelisiniz.")

            if args.action == "hybrid":
                results = hybrid_search(args.question)
            else:
                results = search_chunks(args.question)

            for number, result in enumerate(results, start=1):

                print(f"\n--- Sonuç {number} ---")
                print(f"Sayfa: {result['page_number']}")
                print(f"Kaynak: {result['source_filename']}")

                if args.action == "hybrid":
                    print(f"RRF: {result['rrf_score']:.4f}")
                    print(f"Semantic sıra: {result['semantic_rank']}")
                    print(f"BM25 sıra: {result['bm25_rank']}")
                else:
                    print(f"Benzerlik: {result['score']}")

                print(f"Metin: {result['text']}")

    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Hata: {error}\n")


if __name__ == "__main__":
    main()

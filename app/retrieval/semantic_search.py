
import argparse
import json
import uuid

from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models


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
                "score": round(point.score, 4),
                "page_number": point.payload["page_number"],
                "source_filename": point.payload["source_filename"],
                "text": point.payload["text"],
            }
            for point in results.points
        ]

    finally:
        client.close()


def main() -> None:

    parser = argparse.ArgumentParser(
        description="MedIntel AI Anlamsal Arama"
    )

    parser.add_argument(
        "action",
        choices=["index", "search"],
    )

    parser.add_argument("question", nargs="?")

    args = parser.parse_args()

    try:
        if args.action == "index":
            total = index_chunks()
            print(f"Başarılı! {total} metin parçası indekslendi.")

        elif args.action == "search":
            if not args.question:
                raise ValueError("Arama sorusu girmelisiniz.")

            results = search_chunks(args.question)

            for number, result in enumerate(results, start=1):
                print(f"\n--- Sonuç {number} ---")
                print(f"Benzerlik: {result['score']}")
                print(f"Sayfa: {result['page_number']}")
                print(f"Kaynak: {result['source_filename']}")
                print(f"Metin: {result['text']}")

    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Hata: {error}\n")


if __name__ == "__main__":
    main()

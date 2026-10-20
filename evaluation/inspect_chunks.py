import argparse
import json

from pathlib import Path


CHUNKS_PATH = Path("data/processed/chunks.jsonl")


def inspect_chunks(keyword: str) -> None:

    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError("Chunk dosyası bulunamadı.")

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    matches = [
        chunk
        for chunk in chunks
        if keyword.casefold() in chunk["text"].casefold()
    ]

    print(f"\nEşleşen parça sayısı: {len(matches)}")

    for chunk in matches:
        print(f"\nChunk ID: {chunk['chunk_id']}")
        print(f"Sayfa: {chunk['page_number']}")
        print(f"Metin: {chunk['text']}")
        print("-" * 60)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="MedIntel AI Chunk İnceleme"
    )

    parser.add_argument("keyword")

    args = parser.parse_args()

    try:
        inspect_chunks(args.keyword)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Hata: {error}\n")


if __name__ == "__main__":
    main()
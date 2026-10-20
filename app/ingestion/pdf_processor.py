
import argparse
import hashlib
import json

from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("Parça boyutu sıfırdan büyük olmalıdır.")

    if not 0 <= overlap < chunk_size:
        raise ValueError("Geçersiz örtüşme (overlap) değeri.")

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start
        length = 0

        while end < len(words):
            next_length = len(words[end]) + (1 if end > start else 0)

            if end > start and length + next_length > chunk_size:
                break

            length += next_length
            end += 1

        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        next_start = end
        overlap_length = 0

        while next_start > start:
            word_length = len(words[next_start - 1])
            extra = word_length + (1 if overlap_length else 0)

            if overlap_length + extra > overlap:
                break

            overlap_length += extra
            next_start -= 1

        start = next_start if next_start > start else end

    return chunks


def process_pdf(
    pdf_path: Path,
    output_path: Path,
) -> int:

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF dosyası bulunamadı: {pdf_path}"
        )

    try:
        reader = PdfReader(str(pdf_path))
    except (PdfReadError, OSError) as error:
        raise ValueError("PDF dosyası açılamadı.") from error

    if reader.is_encrypted:
        raise ValueError("Şifreli PDF dosyaları desteklenmiyor.")

    document_id = hashlib.sha256(
        pdf_path.read_bytes()
    ).hexdigest()[:16]

    records = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        chunks = chunk_text(text)

        for chunk_number, chunk in enumerate(chunks, start=1):

            records.append({
                "document_id": document_id,
                "chunk_id": f"{document_id}-{page_number}-{chunk_number}",
                "source_filename": pdf_path.name,
                "page_number": page_number,
                "text": chunk,
            })

    if not records:
        raise ValueError(
            "PDF dosyasında okunabilir metin bulunamadı. "
            "Taranmış PDF dosyaları için OCR gerekebilir."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    return len(records)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="MedIntel AI PDF İşleme Aracı"
    )

    parser.add_argument("pdf", type=Path)

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/chunks.jsonl"),
    )

    args = parser.parse_args()

    try:
        total = process_pdf(args.pdf, args.output)

        print(f"Başarılı! Toplam {total} metin parçası oluşturuldu.")
        print(f"Dosya kaydedildi: {args.output}")

    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Hata: {error}\n")


if __name__ == "__main__":
    main()

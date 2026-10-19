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

    text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

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

    # dosyaya aslında tanımladığımız bir kimlik no eğer dosya adı veya dosya içeriği değişirse bu da değişir
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
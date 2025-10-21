from pypdf import PdfReader
import pytesseract
import re
from pdf2image import convert_from_path
from pathlib import Path
import json

def pdf_to_text(pdf_path: Path):
    """
    Turn pdf input into text.
    :param pdf_path: path to pdf
    :return: text and page number
    """

    reader = PdfReader(str(pdf_path))
    num_pages = len(reader.pages)

    for i in range(num_pages):
        page_num = i + 1
        page = reader.pages[i]
        text = (page.extract_text() or "").strip()

        if len(text) < 40:
            try:
                pil_images = convert_from_path(str(pdf_path), first_page=page_num, last_page=page_num, dpi=300)

                if pil_images:
                    ocr_text = pytesseract.image_to_string(pil_images[0], lang="eng") or ""
                    text = (text + "\n" + ocr_text).strip()

            except Exception:
                pass

        text = clean_text(text)

        yield page_num, text


def clean_text(s: str):
    """
    Clean up and strip text.
    :param s: String
    :return: Cleaned String
    """

    s = re.sub(r'\s+', ' ', s).strip()

    return s


def chunk_by_tokens(text, tokenizer, max_tokens=150, overlap=30):
    """
    Section off chucks of text by number of tokens.
    :param text: Text from pdf.
    :param tokenizer: Tokenizer.
    :param max_tokens: Max number of tokens per chunk.
    :param overlap: Amount of overlap between chunks.
    :return: List of chunks.
    """

    ids = tokenizer(text, truncation=False, add_special_tokens=False, return_attention_mask=False,
                    return_token_type_ids=False)["input_ids"]

    if not (0 <= overlap < max_tokens):
        raise ValueError("overlap must be < max_tokens and >= 0")

    if not ids:

        return []

    chunks, step = [], max_tokens - overlap

    for start in range(0, len(ids), step):
        piece = ids[start:start + max_tokens]
        chunks.append(tokenizer.decode(piece, skip_special_tokens=True))

        if start + max_tokens >= len(ids):
            break

    return chunks


def write_chunks(pdf_path, tokenizer, out_path="/chunks.jsonl") -> Path:
    """
    Write chunks to json file.
    :param pdf_path: Path of pdf file.
    :param tokenizer: Tokenizer used in chunking process.
    :param out_path: path of file chunks being written to, default = "data/chunks.jsonl".
    :return: Path of file chunks being written to, default = "data/chunks.jsonl".
    """

    pdf_path = Path(pdf_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    name = pdf_path.stem

    n_pages = 0
    n_chunks = 0

    with out_path.open("w", encoding="utf-8") as f:

        for slide_id, page_text in pdf_to_text(pdf_path):
            n_pages += 1
            chunks = chunk_by_tokens(page_text, tokenizer)

            for i, ch in enumerate(chunks):
                rec = {"doc_name": name, "slide_id": slide_id, "chunk_id": f"{slide_id}-{i}", "text": ch}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chunks += 1

    print(f"Wrote {n_chunks} chunks from {n_pages} pages -> {out_path}")

    return out_path

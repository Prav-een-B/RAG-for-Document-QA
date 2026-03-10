from unstructured.partition.pdf import partition_pdf
import re
from tqdm import tqdm

def is_bad_chunk(text):

    text = text.strip()

    # very short
    if len(text) < 50:
        return True

    # mostly numbers
    if re.fullmatch(r"[0-9\s\.]+", text):
        return True

    # references / URLs
    if "http://" in text or "https://" in text:
        return True

    # bibliography style lines
    if re.search(r"\b(OECD|ISBN|DOI|Retrieved)\b", text):
        return True

    # repeated numbers like 502502
    if re.fullmatch(r"\d{4,}", text):
        return True

    return False

def ingest_document(pdf_path):
    elements = partition_pdf(
        filename=pdf_path,
        strategy="auto",
        chunking_strategy="by_title",
        max_characters=1200,
        new_after_n_chars=800,
        combine_text_under_n_chars=200,
        languages=["eng"]
    )

    processed = []

    for i, el in tqdm(enumerate(elements), desc="Processing elements", total=len(elements)):

        if not hasattr(el, "text"):
            continue

        text = el.text.strip()
        if is_bad_chunk(text):
            continue

        # Identify type
        if el.category == "Table":
            content_type = "table"
            text = f"Table data: {el.text}"

        elif el.category == "Figure":
            content_type = "chart"
            text = f"Chart description: {el.text}"

        else:
            content_type = "text"
            text = el.text

        processed.append({
            "chunk_id": f"doc_{i:05d}",
            "type": content_type,
            "text": text.strip(),
            "metadata": {
                "page": el.metadata.page_number if el.metadata else None
            }
        })

    return processed
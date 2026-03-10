import numpy as np


def embed_chunks(chunks, model, batch_size=64):
    texts = [c["text"] for c in chunks]

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    for c, v in zip(chunks, vectors):
        c["embedding"] = v.tolist()

    return chunks
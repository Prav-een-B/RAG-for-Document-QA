import numpy as np

def build_matrix(embedded_chunks):
    return np.array([c["embedding"] for c in embedded_chunks])

def retrieve(query, model, embedded_chunks, matrix, k=3):
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    scores = matrix @ q_emb
    top_idx = scores.argsort()[-k:][::-1]

    return [embedded_chunks[i] for i in top_idx]
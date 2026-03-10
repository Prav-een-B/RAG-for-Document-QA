import os
import warnings
import logging

warnings.filterwarnings("ignore")

logging.getLogger().setLevel(logging.ERROR)

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import json
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

from app.ingestion import ingest_document
from app.embedder import embed_chunks
from app.retriever import retrieve
from app.generator import generate_answer

DOC_PATH = "Your_PATH" # ADD YOUR DOCUMENT PATH HERE

CACHE_DIR = "cache"
EMB_FILE = os.path.join(CACHE_DIR, "embeddings.npy")
CHUNK_FILE = os.path.join(CACHE_DIR, "chunks.json")
HASH_FILE = os.path.join(CACHE_DIR, "doc_hash.txt")

os.makedirs(CACHE_DIR, exist_ok=True)

def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

doc_hash = file_hash(DOC_PATH)

use_cache = False

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

if os.path.exists(EMB_FILE) and os.path.exists(CHUNK_FILE) and os.path.exists(HASH_FILE):
    with open(HASH_FILE, "r") as f:
        saved_hash = f.read()

    if saved_hash == doc_hash:
        use_cache = True

if use_cache:
    print("Loading cached embeddings...")
    EMB_MATRIX = np.load(EMB_FILE)
    with open(CHUNK_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
else:
    print("Reading document...")

    chunks = ingest_document(DOC_PATH)

    print("Embedding chunks...")

    embedded = embed_chunks(chunks, model)

    EMB_MATRIX = np.array([e["embedding"] for e in embedded])

    print("Saving cache...")

    np.save(EMB_FILE, EMB_MATRIX)

    chunks_to_save = [
        {
            "chunk_id": e["chunk_id"],
            "text": e["text"],
            "metadata": e.get("metadata", {})
        }
        for e in embedded
    ]
    with open(CHUNK_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks_to_save, f)

    with open(HASH_FILE, "w") as f:
        f.write(doc_hash)

    chunks = chunks_to_save

print("Total chunks:", len(chunks))


while True:
    query = input("\nAsk question (type 'EXIT' to quit): ")
    if query.upper() == "EXIT":
        break
    retrieved = retrieve(query, model, chunks, EMB_MATRIX, k=10)
    pairs = [(query, c["text"]) for c in retrieved]

    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(scores, retrieved),
        key=lambda x: x[0],
        reverse=True
    )

    reranked_chunks = [c for _, c in reranked[:3]]
    print("\nRetrieved context:\n")
    for r in reranked_chunks:
        print("-", r["text"], "\n")

    answer = generate_answer(query, reranked_chunks)
    print("\n------------ ANSWER_SUMMARY ------------:\n", answer)

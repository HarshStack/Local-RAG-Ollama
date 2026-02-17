# ingest_local.py (FINAL WINDOWS-SAFE VERSION)
import os
import pickle
import numpy as np
import faiss
import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def load_pdfs(folder="data"):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            pdf = fitz.open(path)
            for page_num, page in enumerate(pdf):
                text = page.get_text()
                docs.append({"text": text, "source": file, "page": page_num})
    return docs

def split_text(text, chunk=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def build_index():
    os.makedirs("faiss_index", exist_ok=True)

    raw_docs = load_pdfs()
    all_chunks = []
    metadata = []

    for doc in raw_docs:
        parts = split_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for p in parts:
            all_chunks.append(p)
            metadata.append(doc)

    print(f"PDF pages loaded: {len(raw_docs)}")
    print(f"Total text chunks: {len(all_chunks)}")

    # Embeddings
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = embedder.encode(all_chunks, convert_to_numpy=True).astype("float32")

    # FAISS
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    faiss.write_index(index, "faiss_index/index.faiss")

    # BM25
    tokenized = [t.split() for t in all_chunks]
    bm25 = BM25Okapi(tokenized)

    with open("faiss_index/metadata.pkl", "wb") as f:
        pickle.dump({"chunks": all_chunks, "meta": metadata, "bm25": bm25}, f)

    print("✔ FAISS + BM25 index built successfully!")

if __name__ == "__main__":
    build_index()
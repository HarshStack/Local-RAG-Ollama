# chat_local.py
import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()

TOP_K = int(os.getenv("TOP_K", 3))
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

class Streamer(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        print(token, end="", flush=True)

def load_index():
    index = faiss.read_index("faiss_index/index.faiss")
    with open("faiss_index/metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    # match ingest_local.py
    return index, meta["chunks"], meta["meta"], meta["bm25"]

def hybrid_search(query, index, texts, metadatas, bm25, embed, k=TOP_K):
    # BM25
    scores = bm25.get_scores(query.split())
    bm_ids = np.argsort(scores)[::-1][:k]

    # FAISS dense
    q_vec = embed.encode([query], convert_to_numpy=True).astype("float32")
    _, fa_ids = index.search(q_vec, k)
    fa_ids = fa_ids[0]

    # Merge unique
    final_ids = list(set(bm_ids.tolist() + fa_ids.tolist()))

    return [{
        "text": texts[i],
        "meta": metadatas[i]
    } for i in final_ids][:k]

def chat_loop():
    index, texts, metadatas, bm25 = load_index()
    embed = SentenceTransformer("all-MiniLM-L6-v2")

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        streaming=True,
        callbacks=[Streamer()],
        keep_alive="5m"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer ONLY using the provided context."),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])

    chain = prompt | llm | StrOutputParser()

    print("\n🚀 Local RAG Ready (FAISS + BM25 + Ollama)\n")
    while True:
        q = input("\nQ: ").strip()
        if q.lower() == "exit":
            break

        docs = hybrid_search(q, index, texts, metadatas, bm25, embed)
        context = "\n\n".join(
            f"[{d['meta']['source']}] {d['text']}" for d in docs
        )

        print("\n--- Answer ---\n")
        chain.invoke({"question": q, "context": context})

if __name__ == "__main__":
    chat_loop()
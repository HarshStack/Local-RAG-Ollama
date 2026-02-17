<h1 align="center"> Local RAG with Ollama</h1>
<h2>FAISS + BM25 + Sentence Transformers</h2>

A fast, private, and fully–local Retrieval‑Augmented Generation (RAG) system built with:

-Ollama for local LLM inference.
-Sentence‑Transformers for lightning‑fast embeddings
-FAISS for dense vector search

BM25 for keyword / exact‑match search

PyMuPDF for PDF extraction

Hybrid Retrieval (FAISS + BM25 fusion) for highly accurate answers

Streaming responses for instant feedback


No cloud APIs. No Pinecone. No OpenAI.

Everything runs offline on your machine.


✨ Why This Project Exists

Most RAG systems depend on cloud services (OpenAI Embeddings, Pinecone, Weaviate, etc.), which:

❌ cost money

❌ require API keys

❌ need internet

❌ raise privacy concerns (especially for medical/legal/use‑case-sensitive docs)


This project solves that:

✔ 100% offline

✔ No external API calls

✔ No data leaves your machine

✔ Super fast (1–3 seconds end‑to‑end)

✔ Works on Windows

✔ Ideal for medical, financial, private PDFs, and more


🧠 What It Does

You can drop any PDF into ./data/ — such as:

medical reports

personal documents

lecture notes

books

legal material

manuals


The system then:

1️⃣ Extracts text from each page
(using PyMuPDF, fast and reliable)

2️⃣ Splits text into overlapping chunks
(great for long documents)

3️⃣ Builds a dense index
using all-MiniLM-L6-v2 (fast, CPU‑friendly)

4️⃣ Builds a sparse BM25 index
for exact term matching (IDs, numbers, abbreviations)

5️⃣ At query time:

FAISS retrieves semantically related chunks

BM25 retrieves exact‑match chunks

Results are merged into a hybrid top‑k set

6️⃣ ChatOllama generates grounded responses

→ Every answer comes directly from your documents

→ And includes source file + page number

🔥 Features

✔ Hybrid Retrieval (Dense + Sparse)

Best of both worlds:

Dense vectors understand meaning

BM25 catches exact phrases, symptoms, values, names, abbreviations

✔ Local LLM via Ollama

Default model: llama3.2:1b
(You can switch to any Ollama model)

✔ Streaming Answers

Responses appear token-by-token for instant interaction.

✔ Zero Cost

No Pinecone,
No OpenAI,
No subscriptions,
No API keys

✔ Privacy-Friendly

Your PDFs stay completely on your device.

✔ Fast Ingestion

Indexes even large PDFs in seconds.

✔ Works Offline

Perfect for secure/air‑gapped systems.


🛠 How It Works (Technical Architecture)
<img width="866" height="442" alt="image" src="https://github.com/user-attachments/assets/c4651ac8-aeb3-4db6-87c4-58e586f2b46c" />

📂 Project Structure
<img width="955" height="531" alt="image" src="https://github.com/user-attachments/assets/08bb699d-fc11-4036-a5c1-6b0b2dc90904" />









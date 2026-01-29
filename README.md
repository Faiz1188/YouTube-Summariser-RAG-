# 🎥 YouTube RAG Q&A Application

A **Retrieval-Augmented Generation (RAG)** based application that allows you to **ask questions from any caption-enabled YouTube video**. The app downloads subtitles, converts them into embeddings, stores them in a vector database, and uses an LLM to answer questions **strictly from the video content**.

---

## 🚀 Features

* 📌 Works with **YouTube captions (auto/manual)**
* 🔍 Uses **semantic search** with vector embeddings
* 🧠 Answers using **RAG (no hallucination)**
* 💾 Persistent **Chroma vector database** per video
* ⚡ Fast inference with **Groq (LLaMA 3.1)**
* 🎛️ Clean **Streamlit UI** with source inspection

---

## 🧠 Architecture Overview

```
YouTube URL
   ↓
Extract Video ID
   ↓
Download Captions (yt-dlp)
   ↓
Clean & Chunk Text
   ↓
Embeddings (HuggingFace)
   ↓
Chroma Vector DB (per video)
   ↓
Retriever (Top-K chunks)
   ↓
LLM (Groq) + RAG Prompt
   ↓
Final Answer
```

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **LangChain**
* **yt-dlp** – caption extraction
* **ChromaDB** – vector storage
* **HuggingFace Embeddings** – `all-MiniLM-L6-v2`
* **Groq LLM** – `llama-3.1-8b-instant`
* **Streamlit** – UI

---

## 📁 Project Structure

```
.
├── rag_pipeline.py        # Core RAG logic (caption → embeddings → answer)
├── app.py                 # Streamlit frontend
├── vector_db/             # Persistent Chroma DB (auto-created)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2️⃣ Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\\Scripts\\activate  # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file or export variables:

```bash
export GROQ_API_KEY="your_groq_api_key"
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Open browser at: **[http://localhost:8501](http://localhost:8501)**

---

## 🧪 How It Works (Step-by-Step)

1. User enters **YouTube URL**
2. Video ID is extracted
3. Captions are downloaded using **yt-dlp**
4. Captions are cleaned (timestamps & noise removed)
5. Text is split into overlapping chunks
6. Chunks are converted into embeddings
7. Embeddings are stored in **ChromaDB**
8. On question:

   * Relevant chunks are retrieved
   * Context is injected into RAG prompt
   * LLM answers **only from context**

---

## 📌 Important Notes

* ❗ Works **only if captions are available**
* ❌ Videos without subtitles will throw a warning
* 💾 Each video has its **own vector database**
* 🔁 Reusing same video avoids re-embedding

---

## 🧾 Example Prompt

> **Question:** What is self-attention?

The system searches captions, retrieves relevant parts, and answers **strictly from the video explanation**.

<img width="1680" height="1050" alt="Screenshot 2026-01-30 at 12 26 17 AM" src="https://github.com/user-attachments/assets/875aa410-2b4f-473a-9b06-7e4703859754" />



---

## 📚 Future Improvements

* 🔹 Multi-language subtitle support
* 🔹 Timestamp-based source highlighting
* 🔹 Chat history (conversational RAG)
* 🔹 PDF / Blog export of answers
* 🔹 Docker deployment

---

## 👨‍💻 Author

**Faiz Ahmad**
Generative AI & RAG Developer

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to contribute!

Happy Learning 🚀

# rag_pipeline.py

import os
import re
import tempfile
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import yt_dlp

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate



# ---------------- CONFIG ----------------
VECTOR_DB_DIR = Path("vector_db")
VECTOR_DB_DIR.mkdir(exist_ok=True)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# ---------------- UTILS ----------------
def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    return parse_qs(parsed.query)["v"][0]


def load_captions_with_ytdlp(video_id: str) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en"],
            "outtmpl": os.path.join(tmp, "%(id)s.%(ext)s"),
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

        vtt_files = [f for f in os.listdir(tmp) if f.endswith(".vtt")]
        if not vtt_files:
            raise ValueError("❌ No captions found for this video")

        vtt_path = os.path.join(tmp, vtt_files[0])

        # clean VTT
        lines = []
        with open(vtt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if (
                    not line
                    or "-->" in line
                    or re.match(r"^\d+$", line)
                    or line.startswith("WEBVTT")
                ):
                    continue
                lines.append(line)

        return lines


# ---------------- MAIN RAG ----------------
def run_rag(youtube_url: str, query: str):
    video_id = extract_video_id(youtube_url)
    db_dir = VECTOR_DB_DIR / video_id
    db_dir.mkdir(exist_ok=True)

    if any(db_dir.iterdir()):
        vectordb = Chroma(
            persist_directory=str(db_dir),
            embedding_function=embeddings
        )

    else:
        captions = load_captions_with_ytdlp(video_id)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        docs = splitter.create_documents(captions)

        vectordb = Chroma.from_documents(
            docs,
            embedding=embeddings,
            persist_directory=str(db_dir)
        )
        vectordb.persist()

    retriever = vectordb.as_retriever(search_kwargs={"k": 8})
    retrieved_docs = retriever.invoke(query)

    context = "\n".join(d.page_content for d in retrieved_docs)

    rag_prompt = PromptTemplate.from_template("""

You are an expert teacher.

Use ONLY the information from the context below to answer the question.
If information is spread across multiple parts, combine it logically.

Explain in DETAIL, step-by-step, in simple language.
You may give examples if they are implied by the context.



Context:

{context}



Question:

{question}




Answer in clear simple language.

""")

    final_prompt = rag_prompt.format(
        context=context,
        question=query
    )

    answer = llm.invoke(final_prompt).content

    return answer, retrieved_docs

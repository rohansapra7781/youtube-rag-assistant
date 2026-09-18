# 🎥 YouTube Video Q&A Assistant (RAG)

An end-to-end Retrieval-Augmented Generation (RAG) application built with **LangChain (LCEL)**, **FAISS**, **Google Gemini**, and **Streamlit**. It extracts transcripts directly from YouTube videos and allows users to query specific video content without processing full contexts repeatedly.

---

## ⚙️ Architecture & Tech Stack

* **Orchestration**: LangChain (LCEL `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`)
* **Vector Store**: FAISS (in-memory similarity search)
* **Embedding Model**: Google `gemini-embedding-001`
* **LLM**: Google `gemini-2.5-flash`
* **Data Ingestion**: `youtube-transcript-api` (v1.2+)
* **Frontend**: Streamlit

---

## 🚀 Key Features

* **Dynamic Video Ingestion**: Paste any standard YouTube URL; automatically resolves video IDs and handles missing caption edge cases (`TranscriptsDisabled`).
* **Multi-Language Support**: Fallback transcript retrieval (e.g., `en`, `hi`).
* **Optimized Token Economics**: Chunks text via `RecursiveCharacterTextSplitter` to bound context window sizes and minimize API latency.
* **Cached Session State**: Uses `@st.cache_resource` so indexing occurs only once per video per session.

---

## 🛠️ Setup & Local Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rohansapra7781/youtube-rag-assistant.git
   cd youtube-rag-assistant
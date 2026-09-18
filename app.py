import os
import re
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="YouTube Q&A Assistant", page_icon="🎥", layout="wide")

def extract_video_id(url: str):
    regex = r"(?:v=|\/|youtu\.be\/|embed\/)([0-9A-Za-z_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

def format_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"

def format_docs_with_timestamps(docs):
    formatted = []
    for doc in docs:
        ts = doc.metadata.get("timestamp", "00:00")
        formatted.append(f"[{ts}] {doc.page_content}")
    return "\n\n".join(formatted)

@st.cache_resource(show_spinner="Building vector database with local embeddings...")
def build_rag_chain_from_docs(raw_docs, video_key: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    api_key = os.environ.get("GOOGLE_API_KEY")
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=api_key)
    parser = StrOutputParser()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant analyzing a YouTube video.
Answer the user's question using only the retrieved context below.
Cite relevant timestamps like [MM:SS] whenever possible.
If the context is insufficient, state that the video does not cover the topic. Always answer in english, no matter what the transcript language is.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])

    parallel_chain = RunnableParallel({
        "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs_with_timestamps),
        "question": RunnableLambda(lambda x: x["question"]),
        "history": RunnableLambda(lambda x: x.get("history", []))
    })

    base_chain = parallel_chain | prompt | model | parser
    return base_chain

if "session_histories" not in st.session_state:
    st.session_state.session_histories = {}

def get_session_history(session_id: str):
    if session_id not in st.session_state.session_histories:
        st.session_state.session_histories[session_id] = ChatMessageHistory()
    return st.session_state.session_histories[session_id]

# Sidebar
with st.sidebar:
    st.header("Settings")
    input_mode = st.radio("Input Source:", ["YouTube URL", "Direct Transcript / Captions"])

    if input_mode == "YouTube URL":
        yt_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        lang_input = st.text_input("Transcript Language(s)", value="en, hi")
        languages = [code.strip() for code in lang_input.split(",") if code.strip()]
        index_btn = st.button("Index Video", type="primary")

        if index_btn and yt_url:
            vid = extract_video_id(yt_url)
            if not vid:
                st.error("Invalid YouTube URL.")
            else:
                try:
                    yt_api = YouTubeTranscriptApi()
                    transcript_obj = yt_api.fetch(vid, languages=languages)
                    raw_docs = [
                        Document(
                            page_content=snippet.text,
                            metadata={"timestamp": format_timestamp(snippet.start)}
                        )
                        for snippet in transcript_obj
                    ]
                    base_chain = build_rag_chain_from_docs(raw_docs, vid)
                    st.session_state.conversational_chain = RunnableWithMessageHistory(
                        base_chain,
                        get_session_history,
                        input_messages_key="question",
                        history_messages_key="history"
                    )
                    st.session_state.active_video = vid
                    st.session_state.chat_history = []
                    st.session_state.session_histories = {}
                    st.success("Video indexed successfully!")
                except Exception as e:
                    st.error("YouTube blocked the cloud IP address. Switch the radio button above to 'Direct Transcript / Captions' to paste the transcript text directly!")

    else:
        manual_text = st.text_area("Paste Transcript / Subtitles:", height=220, placeholder="Paste raw text or captions here...")
        process_text_btn = st.button("Index Transcript", type="primary")

        if process_text_btn and manual_text.strip():
            raw_docs = []
            current_ts = "00:00"
            lines = manual_text.splitlines()
            ts_pattern = re.compile(r"^\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*$")

            buffer = []
            for line in lines:
                line_str = line.strip()
                if not line_str:
                    continue
                match = ts_pattern.match(line_str)
                if match:
                    if buffer:
                        raw_docs.append(Document(page_content=" ".join(buffer), metadata={"timestamp": current_ts}))
                        buffer = []
                    current_ts = match.group(1)
                else:
                    buffer.append(line_str)

            if buffer:
                raw_docs.append(Document(page_content=" ".join(buffer), metadata={"timestamp": current_ts}))

            if not raw_docs:
                raw_docs = [Document(page_content=manual_text, metadata={"timestamp": "00:00"})]

            base_chain = build_rag_chain_from_docs(raw_docs, "manual_input")
            st.session_state.conversational_chain = RunnableWithMessageHistory(
                base_chain,
                get_session_history,
                input_messages_key="question",
                history_messages_key="history"
            )
            st.session_state.active_video = "manual_input"
            st.session_state.chat_history = []
            st.session_state.session_histories = {}
            st.success("Transcript indexed successfully!")

# Main Window
st.title("🎥 YouTube Video Q&A Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversational_chain" not in st.session_state or st.session_state.conversational_chain is None:
    st.info("👈 Index a video or paste a transcript in the sidebar to start asking questions.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_question := st.chat_input("Ask anything about this content..."):
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching context & generating response..."):
                response = st.session_state.conversational_chain.invoke(
                    {"question": user_question},
                    config={"configurable": {"session_id": st.session_state.active_video}}
                )
                st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
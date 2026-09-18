import re
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="YouTube Q&A Assistant", page_icon="🎥", layout="wide")

def extract_video_id(url: str):
    regex = r"(?:v=|\/|youtu\.be\/|embed\/)([0-9A-Za-z_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Cached per video_id + language combination
@st.cache_resource(show_spinner="Fetching transcript and indexing into FAISS...")
def setup_rag_chain(video_id: str, languages: list):
    try:
        yt_api = YouTubeTranscriptApi()
        transcript_obj = yt_api.fetch(video_id, languages=languages)
        transcript = " ".join(snippet.text for snippet in transcript_obj)
    except TranscriptsDisabled:
        return None, "No captions are enabled for this video."
    except Exception as e:
        return None, f"Could not retrieve transcript ({e}). Try adding another language code."

    # 1. Text Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # 2. Embeddings & In-Memory Vector Store
    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # 3. Model & LCEL Pipeline
    model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
    parser = StrOutputParser()

    prompt = PromptTemplate(
        template="""You are a helpful assistant.
Answer only from the provided context.
If the context is insufficient, just say that you do not know.
Convert the answer in English, no matter what is the language of the transcript.

{context}

Question: {question}
""",
        input_variables=["context", "question"]
    )

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })

    main_chain = parallel_chain | prompt | model | parser
    return main_chain, None

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    yt_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    
    # Language input
    lang_input = st.text_input(
        "Transcript Language(s)", 
        value="en, hi", 
        help="Comma-separated language codes in priority order (e.g. 'en', 'hi', 'es')"
    )
    languages = [code.strip() for code in lang_input.split(",") if code.strip()]
    
    index_btn = st.button("Index Video", type="primary")

    if index_btn and yt_url:
        vid = extract_video_id(yt_url)
        if not vid:
            st.error("Invalid YouTube URL. Please provide a valid link.")
        else:
            st.session_state.chat_history = []
            chain, err = setup_rag_chain(vid, languages)
            if err:
                st.error(err)
                st.session_state.rag_chain = None
            else:
                st.session_state.rag_chain = chain
                st.session_state.active_video = vid
                st.success("Video indexed! You can now ask questions below.")

# --- Main Window ---
st.title("🎥 YouTube Video Q&A Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state or st.session_state.rag_chain is None:
    st.info("👈 Paste a YouTube link in the sidebar and click **Index Video** to start.")
else:
    # Render prior conversation
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User question input bar at the bottom
    if user_question := st.chat_input("Ask anything about this video..."):
        # Display user question
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Retrieve and generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching video context..."):
                answer = st.session_state.rag_chain.invoke(user_question)
                st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
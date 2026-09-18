from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

parser = StrOutputParser()

#1. Indexing (loading the transcript)

video_id = "Gfr50f6ZBvo"
try:
    yt_api = YouTubeTranscriptApi()
    transcript_list = yt_api.fetch(video_id, languages=["en"])

    transcript = " ".join(snippet.text for snippet in transcript_list)
    #print(transcript)

except TranscriptsDisabled:
    print("No captions are available for the video.")


# Text splitting

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# Embedding and storing in Vector Store

vector_store = FAISS.from_documents(chunks, embedding_model)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3})


# Prompt
prompt = PromptTemplate(
    template="""You are a helpful assistant.
                Answer only from the provided context.
                If the context is insufficient, just say that you do not know.

                {context}
                Question: {question}
    """,
    input_variables=['context','question'])

question = "Is the topic of aliens discussed in the video? If yes, then what was discussed?"
retrieved_docs= retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({'context':context_text, 'question':question})

# Generation

answer = model.invoke(final_prompt)

# Building the chains:

def format_docs(retriever_docs):
    context_text= "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel({
    'context':retriever | RunnableLambda(format_docs),
    'question':RunnablePassthrough()
})

main_chain = parallel_chain | prompt | model | parser

result = main_chain.invoke('Can you summarize this video?')

print(result)
import os
import time
from dotenv import load_dotenv
from pathlib import Path
from importlib.metadata import metadata

from gitdb.fun import chunk_size
from pandas.io.formats.format import return_docstring
##Pinecone tools to manage and connect vector database
from pinecone import Pinecone, ServerlessSpec

##Handle and splitting pdf using langchain
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

##Langchain work with Pinecone and Google gemini models
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from referencing import retrieval

#these are building blocks for langchains: prompts, outputs
from sentence_transformers import SentenceTransformer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_huggingface import ChatHuggingFace

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
hf_api_key = os.getenv("HF_TOKEN")

if not pinecone_api_key:
    raise ValueError("PINECE_API_KEY environment variable not set")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
if not hf_api_key:
    raise ValueError("HUGGINGFACE_API_KEY environment variable not set")

#Pinecone configuration
PINECONE_INDEX_NAME = "rag-regtech-data"
PINECONE_NAMESPACE = "ns3-rag-regtech"
PINECONE_DIMENSION = 384
PINECONE_METRIC ="cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

#Google AI Configuration
#GOOGLE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
HUGGINGFACEHUB_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GOOGLE_LLM_MODEL = "gemini-2.5-flash"


PDF_DIRECTORY = Path(__file__).parent / "data"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3

# --- Initialization ---

# Initializes Pinecone connection.
print("Initializing services...")

# Initialize Pinecone client
try:
    from pinecone import Pinecone

    pc = Pinecone(api_key=pinecone_api_key)
    print("✅ Pinecone client initialized.")
except Exception as e:
    print(f"❌ Failed to initialize Pinecone client: {e}")
    exit()

# Initialize Google AI Embeddings & LLM.
# Sets up Gemini for text embedding and chat response.
try:
    #from langchain_google_genai import GoogleGenerativeAIEmbeddings


    embeddings = HuggingFaceEmbeddings(model_name = HUGGINGFACEHUB_EMBEDDING_MODEL)

    print(f"✅ HuggingFace Embeddings model ({HUGGINGFACEHUB_EMBEDDING_MODEL}) initialized.")

    llm = ChatGoogleGenerativeAI(model = GOOGLE_LLM_MODEL, google_api_key=google_api_key, temperature = 0.3)
    print(f"✅ Google LLM model ({GOOGLE_LLM_MODEL}) initialized.")

except Exception as e:
    print(f"❌ Failed to initialize Google AI models: {e}")
    exit()


# Connect to the index (required by LangChain vector store)
index = pc.Index(PINECONE_INDEX_NAME)

print(f"✅ Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")

#retreival through symentaic search
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    namespace=PINECONE_NAMESPACE)


retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
print(f"✅ Retriever configured to fetch top {TOP_K_RESULTS} results.")

# Defines how the AI should answer questions

template = """
You are an assistant knowledgeable about Distributed System Design based on the provided context.
Answer the user's question using ONLY the following context. If the answer is not found in the context, state that clearly.
Do not make up information not present in the context.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)


# Helper to format retrieved docs
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


# Example question
question = "What is Kitchen anology?"

# Prepare inputs
inputs = {
    "context": format_docs(retriever.invoke(question)),
    "question": question
}

# Debug prints (optional)
print(inputs)

# Step 1: Apply prompt
prompted = prompt.invoke(inputs)
print(prompted)

# Step 2: Call LLM
response = llm.invoke(prompted)
print(response)

# Step 3: Parse output
answer = StrOutputParser().invoke(response)

print("✅ RAG chain created successfully.\n")

# Final answer
print(answer)

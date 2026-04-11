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
from config import *

import gradio as gr

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

# --- Initialization ---
# Initializes Pinecone connection.
def pincone_index_connection():
    print("Initializing services...")
    index = None

    # Initialize Pinecone client
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        print("✅ Pinecone client initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize Pinecone client: {e}")
        exit()


def llm_embedding_object(LLM_OR_RETRIEVER_OR_EMBEDDING="LLM"):
    ##Initialize Google AI LLM / Embeddings / Retriever based on input.

    try:
        # --- LLM ---
        if LLM_OR_RETRIEVER_OR_EMBEDDING == "LLM":
            llm = ChatGoogleGenerativeAI(model=GOOGLE_LLM_MODEL, google_api_key=google_api_key, temperature=0.3)
            print(f"✅ Google LLM ({GOOGLE_LLM_MODEL}) initialized.")
            return llm

        # --- EMBEDDINGS ---
        if LLM_OR_RETRIEVER_OR_EMBEDDING == "EMBEDDING":
            embeddings = HuggingFaceEmbeddings(model=HUGGINGFACEHUB_EMBEDDING_MODEL)
            print(f"✅ Google Embeddings model ({HUGGINGFACEHUB_EMBEDDING_MODEL}) initialized.")
            return embeddings

        # --- RETRIEVER ---
        if LLM_OR_RETRIEVER_OR_EMBEDDING == "RETRIEVER":
            embeddings = HuggingFaceEmbeddings(model=HUGGINGFACEHUB_EMBEDDING_MODEL)
            print(f"✅ Google Embeddings model ({HUGGINGFACEHUB_EMBEDDING_MODEL}) initialized.")

            # Connect to existing Pinecone index
            vectorstore = PineconeVectorStore.from_existing_index(
                index_name=PINECONE_INDEX_NAME,
                embedding=embeddings,
                namespace=PINECONE_NAMESPACE,
            )

            # Create retriever
            retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

            print(f"✅ Retriever configured to fetch top {TOP_K_RESULTS} results.")
            return retriever

    except Exception as e:
        print(f"❌ Failed to initialize due to exception: {e}")
        return None

# Defines how the AI should answer questions

def prompt_creator():
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
    return prompt


# Helper to format retrieved docs
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def inputs_creator(question):
    # Initialize retriever
    retriever = llm_embedding_object(LLM_OR_RETRIEVER_OR_EMBEDDING="RETRIEVER")

    # Fetch relevant documents
    context_docs = retriever.invoke(question)

    # Format documents into text
    context_text = format_docs(context_docs)

    # Prepare inputs
    inputs = {
        "context": context_text,
        "question": question
    }

    return inputs

def generate_answer(question="What is monitoring in distributed system?"):
    # Prepare inputs (context + question)
    inputs = inputs_creator(question)

    # Create prompt template
    prompt = prompt_creator()

    # Format prompt with inputs
    prompted = prompt.invoke(inputs)

    # Initialize LLM
    llm = llm_embedding_object(LLM_OR_RETRIEVER_OR_EMBEDDING="LLM")

    # Get response from LLM
    response = llm.invoke(prompted)

    print(response)

    # Parse output
    answer = StrOutputParser().invoke(response)

    print(answer)

    print("✅ RAG chain created successfully.")

    return answer

def load_and_index_pdf(file):
    try:
        # --- Load PDF using LangChain's PyPDFLoader ---
        loader = PyPDFLoader(file.name)
        documents = loader.load()

        if not documents:
            return "❌ Failed to load any content from the uploaded PDF."

        # --- Split content into manageable chunks ---
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(documents)

        # --- Embed and upload to Pinecone ---
        embeddings = llm_embedding_object(
            LLM_OR_RETRIEVER_OR_EMBEDDING="EMBEDDING"
        )

        PineconeVectorStore.from_documents(
            chunks,
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=PINECONE_NAMESPACE
        )

        return f"✅ Successfully uploaded {len(chunks)} chunks to Pinecone."

    except Exception as e:
        return f"❌ Error uploading PDF: {e}"

# --- Tab 1: Upload PDF ---
def upload_tab(file):
    if file is None:
        return "⚠️ Please upload a PDF file."
    return load_and_index_pdf(file)


# --- Tab 2: Ask Question ---
def qa_tab(question):
    if not question or question.strip() == "":
        return "⚠️ Please enter a question."
    return generate_answer(question)



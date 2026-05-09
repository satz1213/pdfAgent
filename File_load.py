import os
import time
from dotenv import load_dotenv
from pathlib import Path

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
#from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

#these are building blocks for langchains: prompts, outputs
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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
HUGGINGFACEHUB_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

PDF_DIRECTORY = Path(__file__).parent / "data"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 3


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

# Initialize Huggingface embedding llm.
# Sets up Gemini for text embedding and chat response.
try:
    embeddings = HuggingFaceEmbeddings(model_name = HUGGINGFACEHUB_EMBEDDING_MODEL)

    print(f"✅ Huggingface Embeddings model ({embeddings}) initialized.")

except Exception as e:
    print(f"❌ Failed to initialize Google AI models: {e}")
    exit()


# --- Pinecone Index Setup ---
# Checks if your desired index already exists.


print(f"\nChecking Pinecone index '{PINECONE_INDEX_NAME}'...")

existing_indexes = pc.list_indexes()

if PINECONE_INDEX_NAME not in [idx.name for idx in existing_indexes]:
    print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating...")

    try:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,  # Ensure dimension matches embeddings
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )

        # Wait for index to be ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            print("Waiting for index to be ready...")
            time.sleep(5)

        print(f"✅ Index '{PINECONE_INDEX_NAME}' created successfully.")

    except Exception as e:
        print(f"❌ Failed to create index '{PINECONE_INDEX_NAME}': {e}")
        exit()

else:
    # Optionally verify dimension if index already exists
    index_description = pc.describe_index(PINECONE_INDEX_NAME)

    if index_description.dimension != PINECONE_DIMENSION:
        print(
            f"❌ ERROR: Index '{PINECONE_INDEX_NAME}' exists but has dimension "
            f"{index_description.dimension}, which does not match required "
            f"dimension {PINECONE_DIMENSION} for model '{HUGGINGFACEHUB_EMBEDDING_MODEL}'."
        )
        print("Please delete the index or use an embedding model with matching dimensions.")
        exit()

    print(f"✅ Index '{PINECONE_INDEX_NAME}' already exists and has the correct dimension ({PINECONE_DIMENSION}).")


# Connect to the index (required by LangChain vector store)
index = pc.Index(PINECONE_INDEX_NAME)

print(f"✅ Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")

# Optional: Show stats before loading
print("Index stats before loading:", index.describe_index_stats())

# --- Data Loading and Processing ---

print(f"\nLoading documents from '{PDF_DIRECTORY}'...")

# Validate directory
if not PDF_DIRECTORY.exists() or not PDF_DIRECTORY.is_dir():
    print(f"❌ Error: PDF directory not found at '{PDF_DIRECTORY}'. Please create it and add your PDF files.")
    exit()

try:
    # Load all PDF files in the folder
    loader = DirectoryLoader(
        str(PDF_DIRECTORY),        # Convert Path to string
        glob="*.pdf",              # Match all PDF files
        loader_cls=PyPDFLoader,    # Use PyPDFLoader for each file
        show_progress=True,        # Display progress
        use_multithreading=True    # Speed up loading
    )

    documents = loader.load()

    print(documents)

    if not documents:
        print(f"❌ No PDF documents found in '{PDF_DIRECTORY}'.")
        exit()

    print(f"✅ Loaded {len(documents)} pages from PDF files.")

    # --- Split documents into chunks ---
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    docs_chunks = text_splitter.split_documents(documents)

    print(f"✅ Split {len(documents)} pages into {len(docs_chunks)} chunks.")

except Exception as e:
    print(f"❌ Failed during document loading or splitting: {e}")
    exit()

# --- Vector Store Setup and Population ---

print("\nSetting up Pinecone vector store...")

# Note:
# This step embeds the chunks and uploads them to Pinecone.
# If data is already uploaded, you may skip or handle differently.

try:
    vectorstore = PineconeVectorStore.from_documents(
        docs_chunks,
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        namespace=PINECONE_NAMESPACE,
        # text_key="text"  # Default is 'text', usually no need to change
    )

    print(f"✅ Data loaded and embedded into Pinecone namespace '{PINECONE_NAMESPACE}'.")

    # Give Pinecone a moment to index
    print("Waiting a few seconds for Pinecone to index...")
    time.sleep(5)

    print("Index stats after loading:", index.describe_index_stats())

except Exception as e:
    print(f"❌ Failed to create or update Pinecone vector store: {e}")
    exit()


# Optional extra wait (as shown in your screenshot)
print("Waiting a few seconds for Pinecone to index...")
time.sleep(5)

print("Index stats after loading:", index.describe_index_stats())

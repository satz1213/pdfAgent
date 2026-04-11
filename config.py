from pathlib import Path

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
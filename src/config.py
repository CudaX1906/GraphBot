import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    LANGSMITH_TRACING = "true"
    LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = "langgraph-demo"

    HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    MAX_NEW_TOKENS = 1024
    
    # New configurations for vector store
    EMBEDDING_MODEL :str = "sentence-transformers/all-MiniLM-L6-v1"
    VECTORSTORE_DIR = "./src/rag/.vectordb"

    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

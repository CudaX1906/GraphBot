from langchain.vectorstores import Chroma
from .config import Config
from langchain_huggingface import HuggingFaceEmbeddings

def create_retriever():
    """Creates and returns a Chroma retriever instance"""
    embeddings = HuggingFaceEmbeddings(model_name= Config.EMBEDDING_MODEL)
    return Chroma(
        persist_directory=Config.VECTORSTORE_DIR,
        embedding_function=embeddings
    ).as_retriever()

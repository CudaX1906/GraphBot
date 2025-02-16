import os
import logging
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from ..config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages vector store operations for PDF indexing."""
    
    def __init__(self):
        self.logger = logger
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
        
        self.persist_directory = Config.VECTORSTORE_DIR
        os.makedirs(self.persist_directory, exist_ok=True)
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Creates a new vector store with provided documents."""
        try:
            vector_store = Chroma.from_documents(documents, self.embeddings, persist_directory=self.persist_directory)
            self.logger.info(f"Vector store created at {self.persist_directory}")
            return vector_store
        except Exception as e:
            self.logger.error(f"Error creating vector store: {e}")
            raise
    
    def load_vector_store(self) -> Optional[Chroma]:
        """Loads an existing vector store."""
        try:
            vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            self.logger.info("Vector store loaded successfully")
            return vector_store
        except Exception as e:
            self.logger.error(f"Error loading vector store: {e}")
            return None

def load_pdf_documents(pdf_path: str) -> List[Document]:
    """Loads and extracts documents from a given PDF file."""
    try:
        loader = PyPDFLoader(pdf_path)
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        return []

def main():
    pdf_path = input("Enter the path to your PDF file: ").strip()
    if not os.path.exists(pdf_path) or not pdf_path.lower().endswith('.pdf'):
        print("Invalid PDF path. Please provide a valid PDF file.")
        return
    
    documents = load_pdf_documents(pdf_path)
    if not documents:
        print("No documents extracted from the PDF.")
        return
    
    manager = VectorStoreManager()
    vector_store = manager.create_vector_store(documents)
    
    query = "What is the main topic of this document?"
    results = vector_store.similarity_search(query, k=3)
    
    print("\nSample Retrieval Results:")
    for i, doc in enumerate(results, 1):
        print(f"\nDocument {i}: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")

if __name__ == "__main__":
    main()

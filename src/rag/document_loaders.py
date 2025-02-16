from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os

def load_pdf_document(
    pdf_path: str, 
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError(f"Invalid file type. Expected PDF, got: {pdf_path}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        chunked_docs = text_splitter.split_documents(documents)
        
        for doc in chunked_docs:
            doc.metadata.update({
                'source': pdf_path,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap
            })
        
        return chunked_docs
    except Exception as e:
        print(f"Error loading PDF {pdf_path}: {str(e)}")
        return []
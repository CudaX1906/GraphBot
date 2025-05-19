from langchain_huggingface.llms import HuggingFaceEndpoint
from .config import Config
import logging
from langchain_groq import ChatGroq
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_llm():
    try:
        if not Config.HUGGINGFACEHUB_API_TOKEN:
            raise ValueError("Hugging Face API token is missing")
            
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        return llm
    except Exception as e:
        logger.error(f"Error creating LLM: {str(e)}")
        raise


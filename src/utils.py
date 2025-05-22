from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.rate_limiters import InMemoryRateLimiter

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
            
        # llm = ChatGroq(
        #     model="llama-3.1-8b-instant",
        #     temperature=0,
        #     max_tokens=None,
        #     timeout=None,
        #     max_retries=2,
        # )
        

        rate_limiter = InMemoryRateLimiter(
            requests_per_second=0.1,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )

        llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1",rate_limiter=rate_limiter)


        return llm
    except Exception as e:
        logger.error(f"Error creating LLM: {str(e)}")
        raise


from langchain_huggingface.llms import HuggingFaceEndpoint
from .config import Config
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_llm():
    try:
        if not Config.HUGGINGFACEHUB_API_TOKEN:
            raise ValueError("Hugging Face API token is missing")
            
        return HuggingFaceEndpoint(
            model=Config.MODEL_NAME,
            huggingfacehub_api_token=Config.HUGGINGFACEHUB_API_TOKEN,
            task="text-generation",
            max_new_tokens=Config.MAX_NEW_TOKENS,
            temperature=0.1
        )
    except Exception as e:
        logger.error(f"Error creating LLM: {str(e)}")
        raise


from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from .schemas import QueryDecompose
from langchain_core.output_parsers import PydanticOutputParser
from .utils import create_llm
from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from .retriever import create_retriever
from.config import Config


@tool
def decompose_tool(query:str):
    """
    Decomposes a complex query into sub-queries.

    This tool takes a complex user query and breaks it down into simpler, 
    more manageable sub-queries. This is particularly useful for handling 
    intricate questions that require multiple angles of inquiry to gather 
    comprehensive information.
    """
    parser = PydanticOutputParser(QueryDecompose)
    prompt_template = """
                    Break down the following complex user query into smaller, meaningful, and unique subqueries that each capture a distinct aspect of the original query.

                    User Query:
                    {user_query}

                    {format_instructions}
                    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables="user_query",
        partial_variables={"format_instructions": parser.get_format_instructions()}
    ).format(user_query=query)

    llm  = create_llm()
    response = llm.invoke(prompt)

    return response

@tool
def rerank_tool(query: str, docs: list):
    """
    Reranks a list of retrieved documents based on their relevance to the given query.
    """
    
    retriever = create_retriever()
    reranker = NVIDIARerank(nvidia_api_key=Config.NVIDIA_API_KEY,model="nvidia / nv-rerankqa-mistral-4b-v3")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=retriever
    )

    
    reranked_docs = compression_retriever.compress_documents(query)

    return reranked_docs

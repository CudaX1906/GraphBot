from langsmith import traceable
from .utils import create_llm
from .state import State
from .schemas import QueryClassification,GradeDocuments,ResponseGenerator
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from .tools import decompose_tool,rerank_tool
from langgraph.types import interrupt
from .config import Config
from .retriever import create_retriever
from .shared import client



@traceable(client=client, project_name="agent-demo",name="Query Analysis", run_type="chain")
def query_analysis_node(state: State):
    llm = create_llm()
    parser = PydanticOutputParser(pydantic_object =QueryClassification)
    template = """You are an expert in query analysis. Classify the given user query as either "Simple" or "Complex" based on these criteria:

                - **Simple Query:** Direct, involving a single entity, basic retrieval, no logical conditions, aggregation, or multi-step reasoning.  
                - Example: "What is the capital of France?"  
                - Example: "Find the price of an iPhone 14."  

                - **Complex Query:** Involves multiple conditions, logical operations, aggregations, reasoning, or multiple entities.  
                - Example: "Find all flights from New York to London that are under $500 and have a layover in Paris."  
                - Example: "Show me the trend of Tesla stock prices over the last five years."  

                ### **User Query:**  
                {user_query}  

                {format_instructions}
                """
    prompt = PromptTemplate(
            template=template,
            input_variables=["user_query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        ).format(user_query=state["query"])
     
    llm_response = llm.invoke(prompt)
    output = parser.parse(llm_response.content)

    
    return {"query":state["query"],"query_complexity":output.complexity}

@traceable(client=client, project_name="agent-demo",name="decompose-tool",run_type="chain")
def decompose_node(state: State):
    """
    Decompose complex queries into sub-queries
    """
    query = state["query"]
    decomposed_queries = decompose_tool.invoke(query)
    
    return {
        "query": query,
        "decomposed_queries": decomposed_queries
    }

@traceable(name="rerank-tool",run_type="chain")
def rerank_node(state: State):
    """
    Rerank retrieved documents
    """
    query = state["query"]
    retrieved_docs = state["retrieved_documents"]
       
    reranked_docs = rerank_tool.invoke({
    "query": query,
    "docs": retrieved_docs,
})

    
    return {
        "query": query,
        "retrieved_documents": reranked_docs,
    }

@traceable(client=client, project_name="agent-demo",name="retriever", run_type="chain")
def retriever_node(state: State):
    query = state.get("query", "")
    query_complexity = state.get("query_complexity", "simple")
    retriever = create_retriever()

    if query_complexity == "simple":
        relevant_docs = retriever.get_relevant_documents(query=query)
    
    elif query_complexity == "complex":
        decomposed_queries = state.get("decomposed_queries", [])
        
        if not decomposed_queries:
            relevant_docs = retriever.get_relevant_documents(query=query)
        else:
            relevant_docs = [
                doc for sub_query in decomposed_queries 
                for doc in retriever.get_relevant_documents(query=sub_query)
            ]

    return {
        "query": query,
        "retrieved_documents": relevant_docs,
    }

@traceable(client=client, project_name="agent-demo",name="grading",run_type="chain")
def grading_node(state: State):
    llm = create_llm()
    parser = PydanticOutputParser(pydantic_object=GradeDocuments)
    template = """"You are a grader assessing relevance of a retrieved document to a user question. \n 
                   If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
                   Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
                   
                   Retrieved Documents : {docs}

                   user question : {query}
                    
                   Note : Make Sure you just provide response as Yes or No , No additional Response is required
                   """
    
    prompt = PromptTemplate(
            template=template,
            input_variables=["query","docs"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
            ).format(docs = state["retrieved_documents"],query = state["query"])
    
    response = llm.invoke(prompt)
   
    return {"query":state["query"],"retrieved_documents":state["retrieved_documents"],"document_grade":response.content}
    
@traceable(client=client, project_name="agent-demo",name="response generation",run_type="chain")
def response_generation(state:State):
    llm = create_llm()
    parser = PydanticOutputParser(pydantic_object=ResponseGenerator)
    template  = """
            You are an expert AI assistant tasked with answering queries based on provided documents. Given a user query and a set of relevant documents, generate a concise and accurate response using only the information available in the documents. If the documents do not contain sufficient details, state that explicitly without making up information.

            ### Input:
            - **Query:** {query}
            - **Documents:**
            {docs}

            ### Response Guidelines:
            1. **Answer Directly:** Provide a precise response based on the documents.
            2. **Cite Sources:** If needed, indicate which document(s) support the response.
            3. **No Hallucination:** Do not generate information beyond what is present in the documents. If the query cannot be answered, state: _"The provided documents do not contain sufficient information to answer this query."_
            4. **Concise & Clear:** Keep the response to the point while ensuring completeness.
            """
    prompt = PromptTemplate(
            template=template,
            input_variables=["query","docs"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
            ).format(docs = state[ "retrieved_documents"],query = state["query"])
    
    response = llm.invoke(prompt)
    return response.content

@traceable(client=client, project_name="agent-demo",name="user-review",run_type="chain")
def user_review_node(state:State):
    is_approved = interrupt({
        "response" : state["current_generation"],
        "action" : "please review and approve the response"
    })

    if is_approved:
        return {"current_generation":state["current_generation"]}
    else:
        k = state["k"]
        return {"k":k+2}

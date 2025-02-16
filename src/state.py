from typing import Optional,List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.documents import Document

class State(TypedDict):
    
    query:str
    query_complexity:Optional[str]
    decomposed_queries : Optional[list] = None
    document_grade:Optional[str]
    retrieved_documents: Optional[List[Document]]
    feedback:Optional[str]
    current_generation: Optional[str]
    k : Optional[int] = 4
    




    
    

from langgraph.graph import StateGraph, START, END
from .state import State
from .nodes import (
    query_analysis_node, 
    retriever_node, 
    grading_node, 
    response_generation, 
    user_review_node,
    rerank_node,
    decompose_node
)

def create_rag_graph():
    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node("query_analysis", query_analysis_node)
    graph_builder.add_node("retriever", retriever_node)
    graph_builder.add_node("decompose", decompose_node)
    graph_builder.add_node("document_grading", grading_node)
    graph_builder.add_node("rerank", rerank_node)
    graph_builder.add_node("response_generation", response_generation)
    graph_builder.add_node("user_review", user_review_node)

    
    graph_builder.add_edge(START, "query_analysis")

    
    graph_builder.add_conditional_edges(
        "query_analysis", 
        lambda state: state.get('query_complexity'),  
        {
            "simple": "retriever",
            "complex": "decompose"
        }
    )

    # Flow connections
    graph_builder.add_edge("decompose", "retriever")
    graph_builder.add_edge("retriever", "document_grading")

    # Conditional document grading
    graph_builder.add_conditional_edges(
        "document_grading",
        lambda state: state.get('document_grade', 'no'), 
        {
            "no": "rerank",
            "yes": "response_generation"
        }
    )

    
    graph_builder.add_edge("rerank", "retriever")
    graph_builder.add_edge("response_generation", "user_review")

    
    graph_builder.add_conditional_edges(
        "user_review",
        lambda state: state.get('user_feedback', 'yes'),  
        {
            "yes": END,
            "no": "retriever"
        }
    )

    return graph_builder.compile()

# Create the graph
rag_graph = create_rag_graph()

# Adaptive RAG System with LangGraph

## Overview

This repository implements an advanced Retrieval-Augmented Generation (RAG) system using LangGraph, designed to provide accurate and contextually relevant answers based on document retrieval. The system features dynamic query handling, document evaluation, and adaptive retrieval mechanisms that increase search depth when necessary.

## Key Features

- **Query Complexity Analysis**: Automatically determines query complexity and routes simple or complex queries appropriately
- **Query Decomposition**: Breaks down complex queries into simpler sub-queries for better retrieval
- **Intelligent Document Grading**: Evaluates retrieved documents for relevance before generating responses
- **Adaptive Retrieval**: Dynamically increases retrieval depth (top k) when user feedback indicates unsatisfactory answers
- **Reranking**: Improves document retrieval when initial results are insufficient
- **End-to-End Tracing**: Full observability through LangSmith integration

## Architecture

The system is built as a directed graph using LangGraph, with nodes representing different processing stages and edges defining the flow logic:

```
flowchart TD
    START(START) --> QUERY[query_analysis]
    
    QUERY -->|simple| RETRIEVE[retriever]
    QUERY -->|complex| DECOMPOSE[decompose]
    
    DECOMPOSE --> RETRIEVE
    
    RETRIEVE --> GRADE[document_grading]
    
    GRADE -->|yes| RESPONSE[response_generation]
    GRADE -->|no| RERANK[rerank]
    
    RERANK --> RETRIEVE
    
    RESPONSE --> USER_REVIEW[user_review]
    
    USER_REVIEW -->|yes| END(END)
    USER_REVIEW -->|no| RETRIEVE

```

When user review indicates an unclear response, the system automatically increments the top k parameter by 2 and re-initiates the retrieval process.

## Setup Requirements

### Installation

```bash
pip install -r requirements.txt
```

Dependencies:
- langchain
- langchain_community
- langgraph
- langsmith
- huggingface_hub
- python-dotenv
- typing-extensions
- chromadb
- pypdf
- langchain_nvidia_ai_endpoints
- langchain_huggingface

### Environment Configuration

Create a `.env` file with the following variables:

```
LANGSMITH_API_KEY=your_langsmith_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
NVIDIA_API_KEY=your_nvidia_api_key
```

## Configuration

The system uses the following configuration:

```python
ANGSMITH_TRACING = "true"
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
LANGSMITH_PROJECT = "langgraph-demo"

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MAX_NEW_TOKENS = 1024

# Vector store configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v1"
VECTORSTORE_DIR = "./src/rag/.vectordb"
```

## Usage

1. Ensure your document corpus is indexed in the vector store
2. Run the RAG system:

```python
from rag.graph import create_rag_graph

# Initialize the graph
rag_graph = create_rag_graph().get_graph()

# Run a query
result = rag_graph.invoke({
    "query": "Your question here"
})
```

## Monitoring and Debugging

The system integrates with LangSmith for comprehensive tracing and monitoring. To view traces:

1. Ensure your `LANGSMITH_API_KEY` is set in the environment
2. Access the LangSmith dashboard to view execution traces in the "langgraph-demo" project



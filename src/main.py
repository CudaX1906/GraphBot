import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
from .shared import client as shared_client
from langsmith import Client
from langgraph.types import Command

import os



load_dotenv()

def main():
    logger.info("Starting LangGraph Chatbot")

    global shared_client
    shared_client = Client(api_key = os.getenv("LANGSMITH_API_KEY"),api_url=os.getenv("LANGSMITH_ENDPOINT"))

    from .graph import rag_graph
    config = {"configurable": {"thread_id": "chatbot"}}
    while True:
        try:
            user_query = input("\nUser: ").strip()
            if user_query.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break

            if not user_query:
                print("Please enter a query.")
                continue

            logger.info(f"Processing query: {user_query}")
            try:
                
                result = rag_graph.invoke({"query": user_query,"k":2},config=config)
                print("="*100)
                print(result["__interrupt__"])
                user_input = input("Approve? (yes/no): ").strip().lower()   
                print(rag_graph.invoke(Command(resume=user_input),config=config))
                print("="*100)
                print("\nAssistant:", result or "Couldn't generate a response.")
                print("=" * 50)
            except Exception as e:
                print("Error:",e)
            

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print("An error occurred. Please try again.")

if __name__ == "__main__":
    logger.info("RAG system initialized")
    main()

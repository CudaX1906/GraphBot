import logging
from .graph import rag_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting LangGraph Chatbot")

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
            result = rag_graph.invoke({"query": user_query})

            print("\nAssistant:", result or "Couldn't generate a response.")
            print("=" * 50)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print("An error occurred. Please try again.")

if __name__ == "__main__":
    logger.info("RAG system initialized")
    main()

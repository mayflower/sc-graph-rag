"""
Example script demonstrating the GraphRAG implementation.
"""
from dotenv import load_dotenv
from graphrag_implementation import GraphRAGSystem

# Load environment variables
load_dotenv()

def main():
    """
    Main function to demonstrate GraphRAG functionality.
    """
    print("Initializing GraphRAG system...")
    graph_rag = GraphRAGSystem(llm_model="gpt-4o")

    print("\nLoading documents and structured data...")
    graph_rag.load_documents("./company_documents/")
    graph_rag.load_structured_data("./product_catalog.csv")

    print("\nExtracting entities and relationships from documents...")
    graph_rag.extract_entities_from_documents()

    print("\nVisualizing the knowledge graph...")
    graph_rag.visualize_graph(save_path="knowledge_graph.png")
    print("Knowledge graph visualization saved to 'knowledge_graph.png'")

    # Example queries
    queries = [
        "What are the key features of the Premium Health Monitor and its related products?",
        "How does the Sleep Analysis Wearable compare to other healthcare monitoring devices?",
        "What wellness products are available for stress management?",
        "Which enterprise solutions offer healthcare provider integration?",
        "What are the benefits of the Remote Patient Monitoring System?"
    ]

    print("\nQuerying the GraphRAG system with example questions:")
    for i, query in enumerate(queries, 1):
        print(f"\n\nQuery {i}: {query}")
        print("-" * 80)
        answer = graph_rag.query(query)
        print(f"Answer: {answer}")

    print("\nGraphRAG demonstration completed!")

if __name__ == "__main__":
    main()

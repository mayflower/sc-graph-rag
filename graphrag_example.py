"""
Example script demonstrating the GraphRAG implementation using the official Microsoft GraphRAG package.
"""
from dotenv import load_dotenv
from graphrag_pipeline import GraphRAGPipeline

# Load environment variables
load_dotenv()

def main():
    """
    Main function to demonstrate GraphRAG functionality.
    """
    pipeline = GraphRAGPipeline()

    # Run indexing
    if pipeline.run_indexing():
        print("Indexing completed successfully.")
    else:
        print("Indexing failed. Please check the logs for details.")
        return

    # Example queries

    global_queries = [
        "What healthcare monitoring devices are available for patients?",
        "How do enterprise health solutions help organizations?",
        "What wellness products are mentioned in the documents?"
    ]

    print("\nQuerying the GraphRAG system with 'global' example questions:")
    for i, query in enumerate(global_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 80)
        answer = pipeline.global_search(query)
        print(f"Answer: {answer}")

    local_queries = [
        "What are the features of the Remote Patient Monitoring System?",
        "What is the Healthcare Provider Integration Suite?",
        "What stress management solutions are available?"
    ]

    print("\nQuerying the GraphRAG system with 'local' example questions:")
    for i, query in enumerate(local_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 80)
        answer = pipeline.local_search(query)
        print(f"Answer: {answer}")

    # Visualize the knowledge graph
    print("\nVisualizing the knowledge graph...")
    pipeline.visualize_graph(output_path="knowledge_graph.png")

    print("\nGraphRAG demonstration completed!")

if __name__ == "__main__":
    main()

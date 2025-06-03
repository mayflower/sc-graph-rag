"""
Example script demonstrating the GraphRAG implementation using the official Microsoft GraphRAG package.
"""
from dotenv import load_dotenv
from pipeline import GraphRAGPipeline
from analyzer import GraphRAGAnalyzer

# Load environment variables
load_dotenv()

def main():
    """
    Main function to demonstrate GraphRAG functionality.
    """
    pipeline = GraphRAGPipeline()
    analyzer = GraphRAGAnalyzer()

    # Run indexing
    if pipeline.run_indexing():
        print("Indexing completed successfully.")
    else:
        print("Indexing failed. Please check the logs for details.")
        return

    # Example queries

    queries = [
        "What are the features of the Remote Patient Monitoring System?",
        "What is the Healthcare Provider Integration Suite?",
        "What stress management solutions are available?"
    ]

    print("\nQuerying the GraphRAG system with example questions:")
    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 80)
        answer = pipeline.search(query)
        print(f"Answer: {answer}")

    print("Analyzing the GraphRAG graph...")

    # Load and display entities
    entities = analyzer.load_entities()
    if entities is not None:
        print("Top 10 entities:")
        print(entities[['title', 'type', 'description']].head(10))

    # Load and display relationships
    relationships = analyzer.load_relationships()
    if relationships is not None:
        print("\nTop 10 relationships:")
        print(relationships[['source', 'target', 'description']].head(10))

    # Graph statistics
    stats = analyzer.analyze_graph_stats()
    if stats:
        print("\nGraph Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")

    print("Visualizing the GraphRAG graph...")
    analyzer.visualize_graph()

    print("\nGraphRAG demonstration completed!")

if __name__ == "__main__":
    main()

# GraphRAG Implementation for Healthcare Product Catalog

This repository contains a GraphRAG (Graph-enhanced Retrieval-Augmented Generation) implementation for a healthcare company's product catalog. The implementation automatically extracts entities and relationships from both structured data (CSV) and unstructured text documents to build a knowledge graph, which is then used to enhance retrieval and answer questions.

## What is GraphRAG?

GraphRAG is an approach that combines the strengths of knowledge graphs with retrieval-augmented generation. It addresses limitations of traditional RAG systems by:

1. Automatically extracting entities and relationships from documents

2. Building a knowledge graph to represent structured information

3. Using the graph structure to enhance retrieval beyond simple vector similarity

4. Integrating graph-based and vector-based retrieval for more comprehensive answers

## How It Works

The GraphRAG system works in the following steps:

1. **Document Loading and Chunking**: Documents are loaded and split into manageable chunks.
2. **Vector Store Creation**: Document chunks are embedded and stored in a vector store (FAISS).
3. **Initial Graph Creation from Structured Data**: A base knowledge graph is created from the structured CSV data.
4. **Entity and Relationship Extraction**: LLMs are used to extract entities and relationships from document chunks.
5. **Graph Enhancement**: The knowledge graph is enhanced with the extracted entities and relationships.

6. **Query Processing**:
   - Entities are extracted from the query
   - Relevant nodes and relationships are retrieved from the graph
   - Relevant documents are retrieved from the vector store
   - Information from both sources is combined to generate a comprehensive answer

## Usage

```python
# Initialize the GraphRAG system
graph_rag = GraphRAGSystem()

# Load documents and structured data
graph_rag.load_documents("./company_documents/")
graph_rag.load_structured_data("./product_catalog.csv")

# Extract entities and relationships from documents
graph_rag.extract_entities_from_documents()

# Visualize the graph
graph_rag.visualize_graph(save_path="knowledge_graph.png")

# Query the system
query = "What are the key features of the Premium Health Monitor and its related products?"
answer = graph_rag.query(query)
```

## Requirements

- Python 3.10+
- OpenAI API key (set in .env file)
- Dependencies: langchain, langchain-openai, networkx, faiss-cpu, matplotlib, pandas

## Installation

1. Ensure you have Python 3.10+ installed
2. Install dependencies:
   ```
   pip install --upgrade pip && pip install pipenv
   pipenv install
   ```
3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the example:
   ```
   pipenv run python graphrag_implementation.py
   ```

## Benefits of This Approach

1. **Automatic Knowledge Graph Construction**: No need to manually define entities and relationships.
2. **Richer Context**: The system can leverage both structured knowledge (graph) and unstructured text (documents).
3. **Better Question Answering**: By understanding the entities in a question and their relationships, the system can provide more accurate and comprehensive answers.
4. **Visualization**: The knowledge graph can be visualized to provide insights into the data structure.
5. **Extensibility**: The system can be easily extended with new documents and data sources.


## References

[Blog Post](https://blog.mayflower.de/)
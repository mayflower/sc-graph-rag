# GraphRAG Implementation for Healthcare Product Catalog

This repository contains a GraphRAG (Graph-enhanced Retrieval-Augmented Generation) implementation for a healthcare company's product catalog using the official Microsoft GraphRAG package. The implementation automatically extracts entities and relationships from unstructured text documents to build a knowledge graph, which is then used to enhance retrieval and answer questions.

## What is GraphRAG?

GraphRAG is an approach that combines the strengths of knowledge graphs with retrieval-augmented generation. It addresses limitations of traditional RAG systems by:

1. Automatically extracting entities and relationships from documents
2. Building a knowledge graph to represent structured information
3. Using the graph structure to enhance retrieval beyond simple vector similarity
4. Integrating graph-based and vector-based retrieval for more comprehensive answers

## How It Works

This implementation uses the official Microsoft GraphRAG CLI to:

1. **Index documents**: The system processes text files in the `input` directory, extracting entities and relationships to build a knowledge graph.
2. **Query the graph**: The system supports both global and local search methods to answer questions about the healthcare products.
3. **Create statistics**: The system generates statistics about the knowledge graph, such as the number of entities and relationships.
4. **Visualize the knowledge graph**: The system generates a visual representation of the entities and relationships in the knowledge graph.

## Project Structure

- `pipeline.py`: The main implementation file that contains the GraphRAGPipeline class for indexing and querying
- `analyzer.py`: Contains the GraphRAGAnalyzer class for analyzing and visualizing the knowledge graph
- `main.py`: Example script demonstrating the GraphRAG functionality
- `input/`: Directory containing the input text files
- `output/`: Directory where GraphRAG stores its output files (entities, relationships, etc., will be created by GraphRAG)
- `logs/`: Directory for log files (will be created by GraphRAG)
- `cache/`: Directory for cached data (will be created by GraphRAG)

## Requirements

- Python 3.10 or higher
- GraphRAG CLI installed and configured
- Dependencies listed in the Pipfile

## Installation

1. Install pipenv if you don't have it
```bash
pip install pipenv
```

2. Install dependencies from Pipfile
```bash
pipenv install
```

## Project Setup

1. Initialize GraphRAG project
```bash
pipenv run graphrag init --root ./
```

2. Add API key (here OpenAI key)
```bash
GRAPHRAG_API_KEY=<API_KEY>
```

3. Approve environment settings
```bash
direnv allow
```

## Usage

Run the example script:
```bash
pipenv run python main.py
```

The script will:
- Run the indexing process
- Execute example search queries
- Generate statistics and visualization of the knowledge graph

## Features

- **CLI-based interaction**: Uses the GraphRAG CLI for indexing and querying
- **Graph analysis**: Provides statistics and insights about the knowledge graph
- **Knowledge graph visualization**: Creates visual representations of entities and relationships

## References

- [Microsoft GraphRAG GitHub Repository](https://github.com/microsoft/graphrag)
- [Microsoft Research Blog Post](https://microsoft.github.io/graphrag/)
- [Blog Post](https://blog.mayflower.de/)
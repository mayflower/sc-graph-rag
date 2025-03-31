import networkx as nx
import pandas as pd
from dotenv import load_dotenv
from langchain.chains import GraphQAChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.graphs import NetworkxEntityGraph
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import matplotlib.pyplot as plt

# Load variables from .env file
load_dotenv()

# 1. Create a knowledge graph from structured data
def create_knowledge_graph(csv_file: str) -> nx.DiGraph:
    """
    Create a knowledge graph from a structured CSV file.

    Args:
        csv_file (str): Path to the CSV file containing product data.

    Returns:
        nx.DiGraph: A directed graph representing the knowledge graph.
    """
    df = pd.read_csv(csv_file, encoding='utf-8')
    graph = nx.DiGraph()

    # Add nodes and edges based on your data structure
    # Example for a product catalog with categories and features
    for _, row in df.iterrows():
        # Add product nodes
        graph.add_node(row['product_id'],
                   type='product',
                   name=row['product_name'],
                   attributes={'price': row['price'], 'launch_date': row['launch_date']})

        # Add category nodes and connect products to categories
        graph.add_node(row['category'], type='category')
        graph.add_edge(row['product_id'], row['category'], relation='belongs_to')

        # Connect related products
        if not pd.isna(row['related_products']):
            related = row['related_products'].split(',')
            for rel in related:
                graph.add_edge(row['product_id'], rel.strip(), relation='related_to')

    return graph

# 2. Set up document retrieval with RAG
def setup_vector_store(documents_dir: str) -> FAISS:
    """
    Set up a vector store for document retrieval.

    Args:
        documents_dir (str): Directory containing text documents.

    Returns:
        FAISS: A vector store for document retrieval.
    """
    # Load documents
    loader = DirectoryLoader(documents_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store

# 3. Combine graph and retrieval for GraphRAG
def setup_graph_rag(knowledge_graph, vector_store):
    """
    Combine knowledge graph and retrieval components.
    """
    # Convert networkx graph to LangChain entity graph
    entity_graph = NetworkxEntityGraph(knowledge_graph)

    # Initialize language model
    llm = OpenAI(temperature=0)

    # Create GraphQA chain
    graph_qa = GraphQAChain.from_llm(
        llm=llm,
        graph=entity_graph,
        verbose=True
    )

    # This is where you would integrate the retrieval component
    # The specific implementation depends on your use case
    # Here's a conceptual approach:

    class GraphRAG:
        """
        A class that combines knowledge graph querying with document retrieval.
        """
        def __init__(self, graph_qa, vector_store):
            """
            Initialize the GraphRAG system.

            Args:
                graph_qa (GraphQAChain): The GraphQA chain for querying the knowledge graph.
                vector_store (FAISS): The vector store for document retrieval.
            """
            self.graph_qa = graph_qa
            self.vector_store = vector_store
            self.llm = llm

        def query(self, question: str) -> str:
            """
            Query the system with a question.

            Args:
                question (str): The question to ask.

            Returns:
                str: The answer to the question.
            """
            # First, try to answer from the knowledge graph
            graph_result = self.graph_qa.invoke(question)

            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(question, k=3)
            doc_content = "\n".join([doc.page_content for doc in docs])

            # Combine graph knowledge with retrieved documents
            combined_prompt = f"""
            Question: {question}
            
            Information from knowledge graph: {graph_result}
            
            Additional information from documents: {doc_content}
            
            Based on ALL the information above, please provide a comprehensive answer.
            """

            final_answer = self.llm.invoke(combined_prompt)
            return final_answer

    return GraphRAG(graph_qa, vector_store)

# 4. Visualize the knowledge graph
def plot_graph(graph: nx.DiGraph):
    """
    Plot the knowledge graph using matplotlib.

    Args:
        graph (nx.DiGraph): The knowledge graph to plot.
    """
    plt.figure(figsize=(12, 8))

    # Increase the 'k' value to spread nodes further apart
    pos = nx.spring_layout(graph, k=0.5, iterations=50)  # Adjust 'k' as needed

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=3000,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        arrowsize=15
    )
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(graph, 'relation')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Knowledge Graph")
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create components
    print("Creating knowledge graph...")
    product_graph = create_knowledge_graph("./product_catalog.csv")

    # Plot the graph if needed
    #print("Plotting the knowledge graph...")
    #plot_graph(product_graph)
    
    print("Setting up vector store...")
    company_vector_store = setup_vector_store("./company_documents/")

    # Setup GraphRAG
    print("Setting up GraphRAG...")
    graph_rag = setup_graph_rag(product_graph, company_vector_store)

    # Query the system
    query = "What are the key features of our premium products in the healthcare category?"
    print("Querying the system...")
    answer = graph_rag.query(query)
    print("Answer:")
    print(answer)

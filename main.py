import os
import networkx as nx
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import GraphQAChain
from langchain.graphs import NetworkxEntityGraph

# Load variables from .env file
load_dotenv()

# 1. Create a knowledge graph from structured data
def create_knowledge_graph(csv_file):
    """Create a knowledge graph from a structured CSV file."""
    df = pd.read_csv(csv_file)
    G = nx.DiGraph()

    # Add nodes and edges based on your data structure
    # Example for a product catalog with categories and features
    for _, row in df.iterrows():
        # Add product nodes
        G.add_node(row['product_id'], 
                   type='product', 
                   name=row['product_name'],
                   attributes={'price': row['price'], 'launch_date': row['launch_date']})

        # Add category nodes and connect products to categories
        G.add_node(row['category'], type='category')
        G.add_edge(row['product_id'], row['category'], relation='belongs_to')

        # Connect related products
        if not pd.isna(row['related_products']):
            related = row['related_products'].split(',')
            for rel in related:
                G.add_edge(row['product_id'], rel.strip(), relation='related_to')

    return G

# 2. Set up document retrieval with RAG
def setup_rag(documents_dir):
    """Set up a retrieval system using document embeddings."""
    # Load documents
    loader = DirectoryLoader(documents_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore

# 3. Combine graph and retrieval for GraphRAG
def setup_graphrag(knowledge_graph, vectorstore):
    """Combine knowledge graph and retrieval components."""
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
        def __init__(self, graph_qa, vectorstore):
            self.graph_qa = graph_qa
            self.vectorstore = vectorstore
            self.llm = llm

        def query(self, question):
            # First, try to answer from the knowledge graph
            graph_result = self.graph_qa.run(question)

            # Retrieve relevant documents
            docs = self.vectorstore.similarity_search(question, k=3)
            doc_content = "\n".join([doc.page_content for doc in docs])

            # Combine graph knowledge with retrieved documents
            combined_prompt = f"""
            Question: {question}
            
            Information from knowledge graph: {graph_result}
            
            Additional information from documents: {doc_content}
            
            Based on ALL the information above, please provide a comprehensive answer.
            """

            final_answer = self.llm(combined_prompt)
            return final_answer

    return GraphRAG(graph_qa, vectorstore)

# Example usage
if __name__ == "__main__":
    # Create components
    graph = create_knowledge_graph("product_catalog.csv")
    vectorstore = setup_rag("./company_documents/")

    # Setup GraphRAG
    graphrag = setup_graphrag(graph, vectorstore)

    # Query the system
    question = "What are the key features of our premium products in the healthcare category?"
    answer = graphrag.query(question)
    print(answer)
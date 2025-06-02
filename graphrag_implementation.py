# Basic imports
import pandas as pd
from dotenv import load_dotenv
import json
import re
from typing import List, Dict, Any

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

# GraphRAG specific imports
import networkx as nx
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

# Visualization imports
import matplotlib.colors as mcolors

# Load environment variables
load_dotenv()

class GraphRAGSystem:
    """
    A GraphRAG implementation that automatically extracts entities and relationships
    from documents to build a knowledge graph, and uses this graph to enhance retrieval.
    """

    def __init__(self, llm_model: str = "gpt-4o"):
        """
        Initialize the GraphRAG system.

        Args:
            llm_model: The LLM model to use for entity extraction and generation
        """
        self.llm = ChatOpenAI(temperature=0, model=llm_model)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.graph = nx.DiGraph()
        self.documents = []
        self.chunks = []
        self.entity_map = {}  # Maps entity names to node IDs

    def load_documents(self, documents_dir: str) -> None:
        """
        Load documents from a directory.

        Args:
            documents_dir: Directory containing text documents
        """
        loader = DirectoryLoader(documents_dir, glob="**/*.txt", loader_cls=TextLoader)
        self.documents = loader.load()
        print(f"Loaded {len(self.documents)} documents")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"Split into {len(self.chunks)} chunks")

        # Create vector store
        self.vector_store = FAISS.from_documents(self.chunks, self.embeddings)
        print("Created vector store")

    def load_structured_data(self, csv_file: str) -> None:
        """
        Load structured data from a CSV file and extract initial entities.

        Args:
            csv_file: Path to the CSV file
        """
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"Loaded structured data with {len(df)} rows")

        # Extract entities from structured data
        for _, row in df.iterrows():
            product_id = row['product_id']
            product_name = row['product_name']
            category = row['category']

            # Add product node
            self._add_entity(product_id, "Product", {
                "name": product_name,
                "price": row['price'],
                "launch_date": row['launch_date']
            })

            # Add category node
            self._add_entity(category, "Category", {})

            # Add relationship between product and category
            self._add_relationship(product_id, category, "belongs_to")

            # Add relationships between related products
            if not pd.isna(row['related_products']):
                related_products = row['related_products'].split(',')
                for rel_product in related_products:
                    self._add_relationship(product_id, rel_product.strip(), "related_to")

    def extract_entities_from_documents(self) -> None:
        """
        Extract entities and relationships from document chunks using LLM.
        """
        entity_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            You are a specialized entity extraction system. Your task is to extract entities and their relationships from the provided text.
            Focus on products, features, benefits, and any other relevant entities.

            Text: {text}

            IMPORTANT: You must respond with ONLY a valid JSON array containing the extracted entities.
            Do not include any explanations, notes, or markdown formatting.

            The JSON structure must be:
            [
                {{
                    "entity": "entity_name",
                    "type": "entity_type",
                    "attributes": {{"attribute1": "value1", "attribute2": "value2"}},
                    "relationships": [
                        {{"related_entity": "related_entity_name", "relation_type": "relation_name"}}
                    ]
                }}
            ]

            If no entities are found, return an empty array: []
            """
        )

        entity_chain = entity_extraction_prompt | self.llm | StrOutputParser()

        print("Extracting entities from documents...")
        for i, chunk in enumerate(self.chunks):
            if i % 10 == 0:
                print(f"Processing chunk {i}/{len(self.chunks)}")

            try:
                # Extract entities from the chunk
                result = entity_chain.invoke({"text": chunk.page_content})

                # Try to extract JSON if it's wrapped in other text
                json_match = re.search(r'\[\s*{.*}\s*\]', result, re.DOTALL)
                if json_match:
                    result = json_match.group(0)

                try:
                    entities = json.loads(result)

                    # Validate the structure
                    if not isinstance(entities, list):
                        print(f"Invalid JSON structure from chunk {i} - not a list")
                        continue

                    # Add entities and relationships to the graph
                    for entity_data in entities:
                        if not isinstance(entity_data, dict):
                            continue

                        entity_name = entity_data.get("entity")
                        entity_type = entity_data.get("type")
                        attributes = entity_data.get("attributes", {})

                        if not entity_name or not entity_type:
                            continue

                        # Add entity to graph
                        self._add_entity(entity_name, entity_type, attributes)

                        # Add relationships
                        for rel in entity_data.get("relationships", []):
                            related_entity = rel.get("related_entity")
                            relation_type = rel.get("relation_type")

                            if related_entity and relation_type:
                                self._add_relationship(entity_name, related_entity, relation_type)

                except json.JSONDecodeError:
                    print(f"Failed to parse JSON from chunk {i}")
                    print(f"Raw result: {result[:100]}...")  # Print first 100 chars for debugging
                    continue

            except Exception as e:
                print(f"Error processing chunk {i}: {str(e)}")
                continue

        print(f"Extracted entities and relationships. Graph now has {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")

    def _add_entity(self, entity_name: str, entity_type: str, attributes: Dict[str, Any]) -> None:
        """
        Add an entity to the graph if it doesn't exist.

        Args:
            entity_name: Name of the entity
            entity_type: Type of the entity
            attributes: Attributes of the entity
        """
        # Create a unique ID for the entity
        entity_id = f"{entity_type}_{entity_name}"

        # Store mapping from entity name to ID
        self.entity_map[entity_name] = entity_id

        # Add node if it doesn't exist
        if entity_id not in self.graph.nodes:
            # Create a clean attributes dictionary
            node_attrs = {
                'name': entity_name,
                'type': entity_type
            }

            # Add other attributes, ensuring they're serializable
            for key, value in attributes.items():
                try:
                    # Convert to string if it's not a basic type
                    if not isinstance(value, (str, int, float, bool, type(None))):
                        node_attrs[key] = str(value)
                    else:
                        node_attrs[key] = value
                except Exception as e:
                    # Skip attributes that can't be processed
                    print(f"Error skipping attribute {key} for entity {entity_name}, {str(e)}")

            # Add the node with attributes
            self.graph.add_node(entity_id, **node_attrs)

    def _add_relationship(self, source_entity: str, target_entity: str, relation_type: str) -> None:
        """
        Add a relationship between two entities.

        Args:
            source_entity: Source entity name
            target_entity: Target entity name
            relation_type: Type of relationship
        """
        # Get entity IDs from the map or create them if they don't exist
        source_id = self.entity_map.get(source_entity)
        target_id = self.entity_map.get(target_entity)

        # If entities don't exist in the map, add them with generic type
        if not source_id:
            self._add_entity(source_entity, "Unknown", {})
            source_id = self.entity_map.get(source_entity)

        if not target_id:
            self._add_entity(target_entity, "Unknown", {})
            target_id = self.entity_map.get(target_entity)

        # Add edge if both nodes exist
        if source_id and target_id:
            self.graph.add_edge(source_id, target_id, relation=relation_type)

    def query(self, question: str, num_results: int = 5) -> str:
        """
        Query the GraphRAG system with a question.

        Args:
            question: The question to ask
            num_results: Number of results to retrieve

        Returns:
            str: The answer to the question
        """
        # Step 1: Extract entities from the question
        entity_extraction_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            You are a specialized entity extraction system. Your task is to extract the main entities mentioned in the following question:

            Question: {question}

            IMPORTANT: Format your response as a comma-separated list of entity names.
            Only output the entity names, nothing else. No explanations, no markdown formatting.

            For example, if the question is "What are the features of the iPhone 16?", your response should be:
            iPhone 16

            If the question is "How does the Tesla Model 3 compare to the BMW i4?", your response should be:
            Tesla Model 3, BMW i4
            """
        )

        entity_chain = entity_extraction_prompt | self.llm | StrOutputParser()
        entities_result = entity_chain.invoke({"question": question})

        # Parse entities
        question_entities = [e.strip() for e in entities_result.split(',') if e.strip()]

        # Step 2: Retrieve relevant nodes from the graph
        graph_results = self._retrieve_from_graph(question_entities)

        # Step 3: Retrieve relevant documents from the vector store
        vector_results = self._retrieve_from_vector_store(question, num_results)

        # Step 4: Combine results and generate answer
        return self._generate_answer(question, graph_results, vector_results)

    def _retrieve_from_graph(self, entities: List[str]) -> Dict[str, Any]:
        """
        Retrieve relevant information from the graph based on entities.

        Args:
            entities: List of entity names to search for

        Returns:
            Dict: Information retrieved from the graph
        """
        graph_info = {
            "entities": [],
            "relationships": []
        }

        # Find nodes that match the entities
        for entity in entities:
            # Try to find the entity in our entity map
            entity_id = self.entity_map.get(entity)

            if entity_id and entity_id in self.graph.nodes:
                # Get node attributes
                node_data = self.graph.nodes[entity_id]

                # Create a clean attributes dictionary
                clean_attributes = {}
                for k, v in node_data.items():
                    if k not in ["name", "type"]:
                        # Convert to string if it's not a basic type
                        if not isinstance(v, (str, int, float, bool, type(None))):
                            clean_attributes[k] = str(v)
                        else:
                            clean_attributes[k] = v

                graph_info["entities"].append({
                    "name": node_data.get("name", entity),
                    "type": node_data.get("type", "Unknown"),
                    "attributes": clean_attributes
                })

                # Get relationships
                for _, target, edge_data in self.graph.out_edges(entity_id, data=True):
                    target_node = self.graph.nodes[target]
                    graph_info["relationships"].append({
                        "source": node_data.get("name", entity),
                        "target": target_node.get("name", target),
                        "relation": edge_data.get("relation", "related_to")
                    })

                # Also get incoming relationships
                for source, _, edge_data in self.graph.in_edges(entity_id, data=True):
                    source_node = self.graph.nodes[source]
                    graph_info["relationships"].append({
                        "source": source_node.get("name", source),
                        "target": node_data.get("name", entity),
                        "relation": edge_data.get("relation", "related_to")
                    })

            # If we can't find the exact entity, try to find similar entities
            else:
                # This is a simple approach - in a real system, you might use embeddings
                # to find semantically similar entities
                for node_id in self.graph.nodes:
                    node_name = self.graph.nodes[node_id].get("name", "")
                    if entity.lower() in node_name.lower():
                        node_data = self.graph.nodes[node_id]

                        # Create a clean attributes dictionary
                        clean_attributes = {}
                        for k, v in node_data.items():
                            if k not in ["name", "type"]:
                                # Convert to string if it's not a basic type
                                if not isinstance(v, (str, int, float, bool, type(None))):
                                    clean_attributes[k] = str(v)
                                else:
                                    clean_attributes[k] = v

                        graph_info["entities"].append({
                            "name": node_data.get("name", node_id),
                            "type": node_data.get("type", "Unknown"),
                            "attributes": clean_attributes
                        })

        return graph_info

    def _retrieve_from_vector_store(self, question: str, num_results: int) -> List[Document]:
        """
        Retrieve relevant documents from the vector store.

        Args:
            question: The question to search for
            num_results: Number of results to retrieve

        Returns:
            List[Document]: Retrieved documents
        """
        if not self.vector_store:
            return []

        return self.vector_store.similarity_search(question, k=num_results)

    def _generate_answer(self, question: str, graph_results: Dict[str, Any], 
                         vector_results: List[Document]) -> str:
        """
        Generate an answer based on graph and vector results.

        Args:
            question: The original question
            graph_results: Results from the graph
            vector_results: Results from the vector store

        Returns:
            str: The generated answer
        """
        # Format graph results
        graph_context = ""
        if graph_results["entities"]:
            graph_context += "Entities:\n"
            for entity in graph_results["entities"]:
                graph_context += f"- {entity['name']} (Type: {entity['type']})\n"
                for attr_name, attr_value in entity['attributes'].items():
                    graph_context += f"  - {attr_name}: {attr_value}\n"

        if graph_results["relationships"]:
            graph_context += "\nRelationships:\n"
            for rel in graph_results["relationships"]:
                graph_context += f"- {rel['source']} {rel['relation']} {rel['target']}\n"

        # Format vector results
        vector_context = ""
        if vector_results:
            vector_context = "\nRelevant Information:\n"
            for _, doc in enumerate(vector_results):
                vector_context += f"{doc.page_content}\n\n"

        # Create the prompt for answer generation
        answer_prompt = PromptTemplate(
            input_variables=["question", "graph_context", "vector_context"],
            template="""
            You are an expert knowledge system with access to a knowledge graph and document repository about healthcare products.

            Answer the following question based on the provided information.

            Question: {question}

            Information from Knowledge Graph:
            {graph_context}

            Information from Documents:
            {vector_context}

            Provide a comprehensive answer that integrates information from both the knowledge graph and the documents.
            Your answer should:
            1. Be factual and based only on the provided information
            2. Integrate structural knowledge from the graph with detailed information from the documents
            3. Prioritize the most specific and relevant details if there's conflicting information
            4. Clearly state if you don't have enough information to answer the question

            Format your answer in a clear, concise manner with proper paragraphs and structure.
            """
        )

        answer_chain = answer_prompt | self.llm | StrOutputParser()
        return answer_chain.invoke({
            "question": question,
            "graph_context": graph_context,
            "vector_context": vector_context
        })

    def visualize_graph(self, figsize=(12, 8), save_path=None):
        """
        Visualize the knowledge graph.

        Args:
            figsize: Figure size
            save_path: Path to save the visualization (if None, display it)
        """
        plt.figure(figsize=figsize)

        # Create a mapping of node types to colors
        node_types = set(nx.get_node_attributes(self.graph, 'type').values())
        color_map = {}

        colors = list(mcolors.TABLEAU_COLORS)
        for i, node_type in enumerate(node_types):
            color_map[node_type] = colors[i % len(colors)]

        # Get node colors based on type
        node_colors = []
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'Unknown')
            node_colors.append(color_map.get(node_type, 'gray'))

        # Use spring layout for visualization
        pos = nx.spring_layout(self.graph, k=0.5, iterations=50)

        # Draw nodes
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            labels={node: self.graph.nodes[node].get('name', node) for node in self.graph.nodes()},
            node_size=3000,
            node_color=node_colors,
            font_size=10,
            font_weight="bold",
            arrowsize=15
        )

        # Draw edge labels
        edge_labels = {(u, v): d.get('relation', '') for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)

        # Add a legend
        legend_elements = [Patch(facecolor=color, label=node_type)
                          for node_type, color in color_map.items()]
        plt.legend(handles=legend_elements, loc='upper right')

        plt.title("Knowledge Graph")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

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

# Example usage
if __name__ == "__main__":
    main()
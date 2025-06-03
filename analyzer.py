import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx

class GraphRAGAnalyzer:
    """
    A class to analyze the output of GraphRAG indexing.
    This class provides methods to load entities, relationships, and communities,
    create a NetworkX graph, analyze graph statistics, and visualize the graph.
    """

    def __init__(self, output_dir="output") -> None:
        """
        Initialize the GraphRAGAnalyzer with the output directory.

        Args:
            output_dir (str): The directory where GraphRAG outputs are stored.
        """
        self.output_dir = Path(output_dir)

    def load_entities(self) -> pd.DataFrame | None:
        """
        Load extracted entities.

        Returns:
            pd.DataFrame: A DataFrame containing entity information, or None if not available.
        """
        entities_file = self.output_dir / "entities.parquet"
        print(f"Loading entities from {entities_file}")
        if entities_file.exists():
            return pd.read_parquet(entities_file)
        return None

    def load_relationships(self) -> pd.DataFrame | None:
        """
        Load extracted relationships.

        Returns:
            pd.DataFrame: A DataFrame containing relationship information, or None if not available.
        """
        relationships_file = self.output_dir / "relationships.parquet"
        if relationships_file.exists():
            return pd.read_parquet(relationships_file)
        return None

    def load_communities(self) -> pd.DataFrame | None:
        """
        Load detected communities.

        Returns:
            pd.DataFrame: A DataFrame containing community information, or None if not available.
        """
        communities_file = self.output_dir / "communities.parquet"
        if communities_file.exists():
            return pd.read_parquet(communities_file)
        return None

    def create_networkx_graph(self):
        """
        Create NetworkX graph from GraphRAG output.

        Returns:
            nx.Graph: A NetworkX graph object containing entities and relationships.
        """
        entities = self.load_entities()
        relationships = self.load_relationships()

        if entities is None or relationships is None:
            return None

        G = nx.Graph()

        # Add nodes
        for _, entity in entities.iterrows():
            G.add_node(entity['title'], 
                      type=entity.get('type', ''),
                      description=entity.get('description', ''))

        # Add edges
        for _, rel in relationships.iterrows():
            G.add_edge(rel['source'], rel['target'],
                      description=rel.get('description', ''),
                      weight=rel.get('weight', 1.0))

        return G

    def analyze_graph_stats(self) -> dict | None:
        """
        Get basic graph statistics.

        Returns:
            dict: A dictionary containing graph statistics such as 
            number of nodes, edges, density, etc.
        """
        G = self.create_networkx_graph()
        if G is None:
            return None

        stats = {
            'Nodes': G.number_of_nodes(),
            'Edges': G.number_of_edges(),
            'Density': nx.density(G),
            'AVG_clustering': nx.average_clustering(G),
            'Connected_components': nx.number_connected_components(G)
        }

        if nx.is_connected(G):
            stats['Diameter'] = nx.diameter(G)
            stats['AVG_path_length'] = nx.average_shortest_path_length(G)

        return stats

    def visualize_graph(self, figsize: tuple=(12, 12)) -> None:
        """
        Visualize the graph using NetworkX and Matplotlib.

        Args:
            figsize: Size of the figure for visualization.
        """
        G = self.create_networkx_graph()
        if G is None:
            print("No graph data available for visualization.")
            return

        plt.figure(figsize=figsize)

        # Create a mapping of node types to colors
        node_types = set(nx.get_node_attributes(G, 'type').values())
        colors = list(mcolors.TABLEAU_COLORS.values())  # Get just the color values
        color_map = {node_type: colors[i % len(colors)] for i, node_type in enumerate(node_types)}

        # Get node colors based on type
        node_colors = [color_map.get(G.nodes[node].get('type', 'Unknown'), 'gray') for node in G.nodes()]

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color=node_colors, font_size=10, font_color='black')

        plt.title("GraphRAG Visualization")
        plt.show()

        # Save the figure
        output_file = self.output_dir / "knowledge_graph.png"
        plt.savefig(output_file)
        print(f"Graph visualization saved to {output_file}")
        plt.close()
        print("Graph visualization completed.")

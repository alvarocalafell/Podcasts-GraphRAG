import networkx as nx
from node2vec import Node2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from neo4j import GraphDatabase
from typing import List, Tuple, Dict, Optional

from src.utils.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, 
    EMBEDDING_DIMENSIONS, WALK_LENGTH, NUM_WALKS, WORKERS,
    EMBEDDINGS_PATH
)
from src.utils.logger import graph_logger as logger

class GraphEmbedder:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.graph: Optional[nx.DiGraph] = None
        self.embeddings: Optional[Dict[str, np.ndarray]] = None

    def create_networkx_graph(self) -> None:
        logger.info("Creating NetworkX graph from Neo4j database...")
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN elementId(n) AS source, labels(n) AS source_labels, n.name AS source_name, n.title AS source_title,
               elementId(m) AS target, labels(m) AS target_labels, m.name AS target_name, m.title AS target_title,
               type(r) AS rel_type
        """
        try:
            with self.driver.session() as session:
                result = session.run(query)
                self.graph = nx.DiGraph()
                for record in result:
                    source = str(record['source'])
                    target = str(record['target']) if record['target'] is not None else None
                    
                    source_name = record['source_title'] if 'Episode' in record['source_labels'] else record['source_name']
                    self.graph.add_node(source, neo4j_id=source, 
                                        labels=record['source_labels'], name=source_name)
                    if target:
                        target_name = record['target_title'] if 'Episode' in record['target_labels'] else record['target_name']
                        self.graph.add_node(target, neo4j_id=target, 
                                            labels=record['target_labels'], name=target_name)
                        self.graph.add_edge(source, target, type=record['rel_type'])
            
            logger.info(f"Created NetworkX graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Error creating NetworkX graph: {str(e)}")
            raise

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self.driver.close()


    def generate_embeddings(self) -> None:
        """Generate graph embeddings using Node2Vec."""
        logger.info("Generating graph embeddings...")
        if not self.graph:
            logger.error("NetworkX graph has not been created. Call create_networkx_graph() first.")
            raise ValueError("NetworkX graph has not been created")
        
        try:
            node2vec = Node2Vec(
                self.graph, 
                dimensions=EMBEDDING_DIMENSIONS, 
                walk_length=WALK_LENGTH, 
                num_walks=NUM_WALKS, 
                workers=WORKERS
            )
            model = node2vec.fit(window=10, min_count=1, batch_words=4)
            
            self.embeddings = {self.graph.nodes[node]['neo4j_id']: model.wv[node] for node in self.graph.nodes()}
            logger.info(f"Generated embeddings for {len(self.embeddings)} nodes")
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def get_similar_nodes(self, node_name_or_id: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Find nodes similar to the given node name or ID."""
        logger.info(f"Finding nodes similar to '{node_name_or_id}'...")
        if not self.embeddings:
            logger.error("Embeddings have not been generated. Call generate_embeddings() first.")
            raise ValueError("Embeddings have not been generated")

        try:
            # Find the node ID by name or use the provided ID
            node_id = None
            for n, data in self.graph.nodes(data=True):
                if data['neo4j_id'] == node_name_or_id:
                    node_id = data['neo4j_id']
                    break
                elif data.get('name') == node_name_or_id or data.get('title') == node_name_or_id:
                    node_id = data['neo4j_id']
                    break

            if node_id is None:
                logger.warning(f"Node '{node_name_or_id}' not found in the graph")
                return []

            if node_id not in self.embeddings:
                logger.warning(f"Node '{node_name_or_id}' (ID: {node_id}) not found in embeddings")
                return []

            logger.info(f"Found node '{node_name_or_id}' with ID: {node_id}")

            target_embedding = self.embeddings[node_id]
            similarities = []

            for neo4j_id, embedding in self.embeddings.items():
                if neo4j_id != node_id:
                    similarity = cosine_similarity([target_embedding], [embedding])[0][0]
                    node_data = next((data for n, data in self.graph.nodes(data=True) if data['neo4j_id'] == neo4j_id), None)
                    node_name = node_data.get('name') or node_data.get('title') or f"Unknown (ID: {neo4j_id})"
                    node_type = ', '.join(node_data.get('labels', []))
                    similarities.append((node_name, similarity, node_type))

            top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
            logger.info(f"Found {len(top_similar)} similar nodes")
            return top_similar
        except Exception as e:
            logger.error(f"Error finding similar nodes: {str(e)}")
            raise

    def save_embeddings(self) -> None:
        """Save the generated embeddings to a file."""
        logger.info(f"Saving embeddings to {EMBEDDINGS_PATH}...")
        if not self.embeddings:
            logger.error("No embeddings to save. Generate embeddings first.")
            raise ValueError("No embeddings to save")

        try:
            embeddings_dict = {neo4j_id: emb.tolist() for neo4j_id, emb in self.embeddings.items()}
            np.save(EMBEDDINGS_PATH, embeddings_dict)
            logger.info("Embeddings saved successfully")
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
            raise

    def load_embeddings(self) -> None:
        """Load pre-computed embeddings from a file."""
        logger.info(f"Loading embeddings from {EMBEDDINGS_PATH}...")
        try:
            loaded_dict = np.load(EMBEDDINGS_PATH, allow_pickle=True).item()
            self.embeddings = {str(neo4j_id): np.array(emb) for neo4j_id, emb in loaded_dict.items()}
            logger.info(f"Loaded embeddings for {len(self.embeddings)} nodes")
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise

def main():
    embedder = GraphEmbedder()
    try:
        embedder.create_networkx_graph()
        embedder.generate_embeddings()
        embedder.save_embeddings()

        # Example: Find similar nodes to a given episode
        similar_nodes = embedder.get_similar_nodes("Life 3.0")
        logger.info(f"Nodes similar to 'Life 3.0': {similar_nodes}")

        # Load embeddings in a new session
        new_embedder = GraphEmbedder()
        new_embedder.create_networkx_graph()  # Still need to create the graph structure
        new_embedder.load_embeddings()

        # Use loaded embeddings
        similar_nodes = new_embedder.get_similar_nodes("Consciousness")
        logger.info(f"Nodes similar to 'Consciousness': {similar_nodes}")

    except Exception as e:
        logger.error(f"An error occurred in the main execution: {str(e)}")
    finally:
        embedder.close()
        if 'new_embedder' in locals():
            new_embedder.close()

if __name__ == "__main__":
    main()
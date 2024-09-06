import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
import json

from src.utils.config import (
    GNN_HIDDEN_CHANNELS, GNN_EMBEDDING_DIM, GNN_LEARNING_RATE, 
    GNN_WEIGHT_DECAY, GNN_EPOCHS, GNN_MODEL_PATH, EMBEDDINGS_PATH
)
from src.utils.logger import model_logger as logger
from src.graph.graph_embedder import GraphEmbedder

class GNNModel(torch.nn.Module):
    def __init__(self, num_features: int):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_features, GNN_HIDDEN_CHANNELS)
        self.conv2 = GCNConv(GNN_HIDDEN_CHANNELS, GNN_EMBEDDING_DIM)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class GNNTrainer:
    def __init__(self, embedder: GraphEmbedder):
        self.embedder = embedder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[GNNModel] = None
        self.data: Optional[Data] = None
        self.node_id_map: Dict[str, int] = {}
        self.reverse_node_id_map: Dict[int, str] = {}
        self.node_types: Dict[str, str] = {}
    
    def prepare_data(self) -> None:
        logger.info("Preparing data for GNN...")
        graph = self.embedder.graph
        initial_embeddings = self.embedder.embeddings

        if graph is None or initial_embeddings is None:
            logger.error("Graph or embeddings not found. Ensure GraphEmbedder has been properly initialized.")
            raise ValueError("Graph or embeddings not available")

        # Create a mapping from Neo4j IDs to consecutive integers
        neo4j_ids = [data['neo4j_id'] for _, data in graph.nodes(data=True)]
        self.node_id_map = {neo4j_id: i for i, neo4j_id in enumerate(neo4j_ids)}
        self.reverse_node_id_map = {i: neo4j_id for neo4j_id, i in self.node_id_map.items()}

        # Prepare node features and type information
        node_features = []
        for _, data in graph.nodes(data=True):
            neo4j_id = data['neo4j_id']
            if neo4j_id in initial_embeddings:
                node_features.append(initial_embeddings[neo4j_id])
                self.node_types[neo4j_id] = data['labels'][0]  # Assume first label is the node type
            else:
                logger.warning(f"No embedding found for node {neo4j_id}")

        # Prepare edge index
        edge_index = []
        for e in graph.edges():
            source_neo4j_id = graph.nodes[e[0]]['neo4j_id']
            target_neo4j_id = graph.nodes[e[1]]['neo4j_id']
            if source_neo4j_id in self.node_id_map and target_neo4j_id in self.node_id_map:
                edge_index.append([self.node_id_map[source_neo4j_id], self.node_id_map[target_neo4j_id]])
            else:
                logger.warning(f"Edge {source_neo4j_id} -> {target_neo4j_id} skipped due to missing node in map")

        # Convert to PyTorch tensors
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        self.data = Data(x=x, edge_index=edge_index)
        self.data = self.data.to(self.device)
        logger.info(f"Prepared data with {self.data.num_nodes} nodes and {self.data.num_edges} edges")

    def train_model(self) -> None:
        logger.info("Training GNN model...")
        if self.data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        num_features = self.data.num_node_features
        self.model = GNNModel(num_features).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=GNN_LEARNING_RATE, weight_decay=GNN_WEIGHT_DECAY)

        for epoch in range(GNN_EPOCHS):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            
            # Use a simple reconstruction loss
            loss = F.mse_loss(out, self.data.x)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch {epoch+1}/{GNN_EPOCHS}, Loss: {loss.item():.4f}')

        logger.info("GNN training completed.")
        self.save_model()
        self.save_embeddings()

    def save_model(self) -> None:
        logger.info(f"Saving GNN model to {GNN_MODEL_PATH}")
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        torch.save(self.model.state_dict(), GNN_MODEL_PATH)

    def load_model(self, num_features: int) -> None:
        logger.info(f"Loading GNN model from {GNN_MODEL_PATH}")
        self.model = GNNModel(num_features).to(self.device)
        self.model.load_state_dict(torch.load(GNN_MODEL_PATH))
        self.model.eval()

    def get_embeddings(self) -> Dict[str, np.ndarray]:
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train_model() or load_model() first.")
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
        embeddings = {self.reverse_node_id_map[i]: emb.cpu().numpy() 
                      for i, emb in enumerate(out)}
        return embeddings

    def save_embeddings(self) -> None:
        logger.info(f"Saving learned embeddings to {EMBEDDINGS_PATH}")
        embeddings = self.get_embeddings()
        embeddings_dict = {node_id: emb.tolist() for node_id, emb in embeddings.items()}
        with open(EMBEDDINGS_PATH, 'w') as f:
            json.dump(embeddings_dict, f)

    def load_embeddings(self) -> Dict[str, np.ndarray]:
        logger.info(f"Loading learned embeddings from {EMBEDDINGS_PATH}")
        with open(EMBEDDINGS_PATH, 'r') as f:
            embeddings_dict = json.load(f)
        return {node_id: np.array(emb) for node_id, emb in embeddings_dict.items()}

    def find_similar_nodes(self, node_id: str, top_k: int = 5) -> List[Dict[str, any]]:
        embeddings = self.get_embeddings()
        if node_id not in embeddings:
            logger.warning(f"Node {node_id} not found in embeddings.")
            return []

        target_embedding = embeddings[node_id]
        similarities = []
        for other_id, other_embedding in embeddings.items():
            if other_id != node_id:
                sim = cosine_similarity([target_embedding], [other_embedding])[0][0]
                similarities.append((other_id, sim, self.node_types[other_id]))

        top_similar = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        return [{"id": id, "similarity": sim, "type": type} for id, sim, type in top_similar]

def main():
    # Initialize GraphEmbedder and prepare the graph and embeddings
    embedder = GraphEmbedder()
    embedder.create_networkx_graph()
    embedder.load_embeddings()  # Assuming embeddings are already generated and saved

    # Initialize and use GNNTrainer
    trainer = GNNTrainer(embedder)
    trainer.prepare_data()
    trainer.train_model()

    # Example: Find similar nodes
    example_node = '4:8aa96113-6ff5-49d1-8f53-18c41bf48bc4:460'  # Replace with an actual node ID from your graph
    similar_nodes = trainer.find_similar_nodes(example_node, top_k=5)
    logger.info(f"Nodes similar to {example_node}:")
    for node in similar_nodes:
        logger.info(f"  ID: {node['id']}, Type: {node['type']}, Similarity: {node['similarity']:.4f}")

if __name__ == "__main__":
    main()
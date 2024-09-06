import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data
import numpy as np
from typing import List, Optional, Tuple, Dict
from sklearn.model_selection import train_test_split

from src.utils.config import (
    GNN_HIDDEN_CHANNELS, GNN_NUM_LAYERS, GNN_DROPOUT, GNN_LEARNING_RATE, 
    GNN_WEIGHT_DECAY, GNN_EPOCHS, GNN_MODEL_PATH
)
from src.utils.logger import model_logger as logger
from src.graph.graph_embedder import GraphEmbedder

class GNNModel(torch.nn.Module):
    def __init__(self, num_features: int, num_classes: int, conv_type: str = 'GCN'):
        super(GNNModel, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.conv_type = conv_type

        ConvLayer = {
            'GCN': GCNConv,
            'SAGE': SAGEConv,
            'GAT': GATConv
        }.get(conv_type, GCNConv)

        self.convs.append(ConvLayer(num_features, GNN_HIDDEN_CHANNELS))
        for _ in range(GNN_NUM_LAYERS - 2):
            self.convs.append(ConvLayer(GNN_HIDDEN_CHANNELS, GNN_HIDDEN_CHANNELS))
        self.convs.append(ConvLayer(GNN_HIDDEN_CHANNELS, num_classes))

        self.dropout = torch.nn.Dropout(p=GNN_DROPOUT)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class GNNTrainer:
    def __init__(self, embedder: GraphEmbedder):
        self.embedder = embedder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[GNNModel] = None
        self.data: Optional[Data] = None
        self.node_id_map: Dict[str, int] = {}
        self.reverse_node_id_map: Dict[int, str] = {}
    
    def train_model(self, conv_type: str = 'GCN') -> None:
        logger.info(f"Training GNN model with {conv_type} convolution...")
        if self.data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        num_features = self.data.num_node_features
        num_classes = len(torch.unique(self.data.y))

        self.model = GNNModel(num_features, num_classes, conv_type).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=GNN_LEARNING_RATE, weight_decay=GNN_WEIGHT_DECAY)

        # Split data into train and test sets
        num_nodes = self.data.num_nodes
        train_mask, test_mask = train_test_split(range(num_nodes), test_size=0.2, random_state=42)
        train_mask = torch.tensor(train_mask)
        test_mask = torch.tensor(test_mask)

        best_test_acc = 0.0
        for epoch in range(GNN_EPOCHS):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(self.data.x, self.data.edge_index)
            loss = F.nll_loss(out[train_mask], self.data.y[train_mask])
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                train_acc = self.evaluate(train_mask)
                test_acc = self.evaluate(test_mask)
                logger.info(f'Epoch {epoch+1}/{GNN_EPOCHS}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    logger.info(f"New best test accuracy: {best_test_acc:.4f}")
                    self.save_model()

        logger.info(f"GNN training completed. Best test accuracy: {best_test_acc:.4f}")

    def evaluate(self, mask: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred = out.argmax(dim=1)
            correct = pred[mask] == self.data.y[mask]
            return correct.sum().item() / mask.sum().item()

    def save_model(self) -> None:
        logger.info(f"Saving GNN model to {GNN_MODEL_PATH}")
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        torch.save(self.model.state_dict(), GNN_MODEL_PATH)

    def load_model(self, num_features: int, num_classes: int, conv_type: str = 'GCN') -> None:
        logger.info(f"Loading GNN model from {GNN_MODEL_PATH}")
        self.model = GNNModel(num_features, num_classes, conv_type).to(self.device)
        self.model.load_state_dict(torch.load(GNN_MODEL_PATH))
        self.model.eval()

    def prepare_data(self) -> None:
        logger.info("Preparing data for GNN...")
        graph = self.embedder.graph
        embeddings = self.embedder.embeddings

        if graph is None or embeddings is None:
            logger.error("Graph or embeddings not found. Ensure GraphEmbedder has been properly initialized.")
            raise ValueError("Graph or embeddings not available")

        # Create a mapping from Neo4j IDs to consecutive integers
        neo4j_ids = [data['neo4j_id'] for _, data in graph.nodes(data=True)]
        self.node_id_map = {neo4j_id: i for i, neo4j_id in enumerate(neo4j_ids)}
        self.reverse_node_id_map = {i: neo4j_id for neo4j_id, i in self.node_id_map.items()}

        # Prepare node features and labels
        node_features = []
        node_labels = []
        label_map = {}
        for _, data in graph.nodes(data=True):
            neo4j_id = data['neo4j_id']
            if neo4j_id in embeddings:
                node_features.append(embeddings[neo4j_id])
                
                label = data['labels'][0]  # Using the first label as the main category
                if label not in label_map:
                    label_map[label] = len(label_map)
                node_labels.append(label_map[label])
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
        y = torch.tensor(node_labels, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        self.data = Data(x=x, edge_index=edge_index, y=y)
        self.data = self.data.to(self.device)
        logger.info(f"Prepared data with {self.data.num_nodes} nodes and {self.data.num_edges} edges")

    def predict(self, node_ids: List[str]) -> List[Tuple[str, int]]:
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred = out.argmax(dim=1)
        
        predictions = []
        for node_id in node_ids:
            if node_id in self.node_id_map:
                integer_id = self.node_id_map[node_id]
                prediction = pred[integer_id].item()
                predictions.append((node_id, prediction))
            else:
                logger.warning(f"Node ID {node_id} not found in the graph. Skipping prediction.")
                predictions.append((node_id, None))
        
        return predictions

def main():
    # Initialize GraphEmbedder and prepare the graph and embeddings
    embedder = GraphEmbedder()
    embedder.create_networkx_graph()
    embedder.load_embeddings()  # Assuming embeddings are already generated and saved

    # Initialize and use GNNTrainer
    trainer = GNNTrainer(embedder)
    trainer.prepare_data()
    trainer.train_model(conv_type='GCN')

    # Example: Make predictions for some nodes
    # Use actual Neo4j IDs from your graph
    node_ids = [
        '4:8aa96113-6ff5-49d1-8f53-18c41bf48bc4:460',  # Life 3.0
        '4:8aa96113-6ff5-49d1-8f53-18c41bf48bc4:490',  # Consciousness
        '4:8aa96113-6ff5-49d1-8f53-18c41bf48bc4:513',  # AI in the Age of Reason
        '4:8aa96113-6ff5-49d1-8f53-18c41bf48bc4:533',  # Deep Learning
        '4:8aa96113-6ff5-49d1-8f53-18c41bf48bc4:546'   # Statistical Learning
    ]
    predictions = trainer.predict(node_ids)
    for node_id, pred in predictions:
        if pred is not None:
            logger.info(f"Prediction for node {node_id}: {pred}")
        else:
            logger.warning(f"Unable to make prediction for node {node_id}")

if __name__ == "__main__":
    main()
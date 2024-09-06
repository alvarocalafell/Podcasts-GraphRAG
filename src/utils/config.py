import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Neo4j configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'neo4j+s://d3d951cb.databases.neo4j.io')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'kv4h0oTLOqzpR-knqHRJw8LIWlUxmP0BEY0v27V5dfw')

# Graph embedding configuration
EMBEDDING_DIMENSIONS = 128
WALK_LENGTH = 80
NUM_WALKS = 10
WORKERS = 4

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'toy_podcast_data.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'toy_processed_podcast_data.csv')
EMBEDDINGS_PATH = os.path.join(DATA_DIR, 'embeddings', 'podcast_embeddings.npy')


from neo4j import GraphDatabase
import pandas as pd
import logging
from tqdm import tqdm
import ast
from typing import List, Dict, Any, Optional
from src.utils.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, PROCESSED_DATA_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def run_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            try:
                result = session.run(query, parameters)
                return list(result)
            except Exception as e:
                logger.error(f"Error running query: {e}")
                raise

    def create_constraint(self, label: str, property: str) -> None:
        create_query = f"CREATE CONSTRAINT {label}_{property}_unique IF NOT EXISTS FOR (n:{label}) REQUIRE n.{property} IS UNIQUE"
        try:
            self.run_query(create_query)
            logger.info(f"Created constraint for {label}.{property}")
        except Exception as e:
            logger.error(f"Error creating constraint: {e}")
            raise

    def create_lex_fridman_node(self) -> None:
        query = """
        MERGE (l:Person {name: 'Lex Fridman'})
        SET l:Host
        RETURN l
        """
        try:
            self.run_query(query)
            logger.info("Created Lex Fridman node")
        except Exception as e:
            logger.error(f"Error creating Lex Fridman node: {e}")
            raise

    def create_episode_node(self, row: pd.Series) -> None:
        query = """
        MERGE (e:Episode {title: $title})
        SET e.id = $id
        WITH e
        MATCH (l:Person {name: 'Lex Fridman'})
        MERGE (l)-[:HOSTS]->(e)
        RETURN e
        """
        try:
            self.run_query(query, {"id": str(row['id']), "title": row['title']})
        except Exception as e:
            logger.error(f"Error creating episode node: {e}")
            raise

    def create_guest_node(self, guest: str, episode_title: str) -> None:
        query = """
        MERGE (g:Person {name: $guest})
        SET g:Guest
        WITH g
        MATCH (e:Episode {title: $episode_title})
        MERGE (e)-[:FEATURES]->(g)
        """
        try:
            self.run_query(query, {"guest": guest, "episode_title": episode_title})
        except Exception as e:
            logger.error(f"Error creating guest node: {e}")
            raise

    def create_entity_nodes(self, entities: List[str], episode_title: str) -> None:
        query = """
        UNWIND $entities as entity
        MERGE (p:Person {name: entity})
        WITH p
        MATCH (e:Episode {title: $episode_title})
        MERGE (e)-[:MENTIONS]->(p)
        """
        try:
            self.run_query(query, {"episode_title": episode_title, "entities": entities})
        except Exception as e:
            logger.error(f"Error creating entity nodes: {e}")
            raise

    def create_organization_nodes(self, organizations: List[str], episode_title: str) -> None:
        query = """
        UNWIND $organizations as org
        MERGE (o:Organization {name: org})
        WITH o
        MATCH (e:Episode {title: $episode_title})
        MERGE (e)-[:FEATURES]->(o)
        """
        try:
            self.run_query(query, {"episode_title": episode_title, "organizations": organizations})
        except Exception as e:
            logger.error(f"Error creating organization nodes: {e}")
            raise

    def create_technology_nodes(self, technologies: List[str], episode_title: str) -> None:
        query = """
        UNWIND $technologies as tech
        MERGE (t:Technology {name: tech})
        WITH t
        MATCH (e:Episode {title: $episode_title})
        MERGE (e)-[:DISCUSSES]->(t)
        """
        try:
            self.run_query(query, {"episode_title": episode_title, "technologies": technologies})
        except Exception as e:
            logger.error(f"Error creating technology nodes: {e}")
            raise

    def create_topic_nodes(self, topics: List[str], episode_title: str) -> None:
        query = """
        UNWIND $topics as topic
        MERGE (t:Topic {name: topic})
        WITH t
        MATCH (e:Episode {title: $episode_title})
        MERGE (e)-[:COVERS]->(t)
        """
        try:
            self.run_query(query, {"episode_title": episode_title, "topics": topics})
        except Exception as e:
            logger.error(f"Error creating topic nodes: {e}")
            raise

    def build_graph(self, data: pd.DataFrame) -> None:
        logger.info("Building knowledge graph...")
        try:
            self.create_lex_fridman_node()
            for _, row in tqdm(data.iterrows(), total=len(data), desc="Building Graph"):
                self.create_episode_node(row)
                self.create_guest_node(row['guest'], row['title'])
                self.create_entity_nodes(ast.literal_eval(row['people']), row['title'])
                self.create_organization_nodes(ast.literal_eval(row['organizations']), row['title'])
                self.create_technology_nodes(ast.literal_eval(row['technologies']), row['title'])
                self.create_topic_nodes(ast.literal_eval(row['topics']), row['title'])
            logger.info("Knowledge graph built successfully!")
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            raise

    def delete_all(self) -> None:
        query = "MATCH (n) DETACH DELETE n"
        try:
            self.run_query(query)
            logger.info("All nodes and relationships have been deleted.")
        except Exception as e:
            logger.error(f"Error deleting all nodes and relationships: {e}")
            raise

def main() -> None:
    try:
        # Load processed data
        logger.info("Loading processed data...")
        data = pd.read_csv(PROCESSED_DATA_PATH)
        logger.info(f"Loaded {len(data)} rows of processed data")

        # Initialize knowledge graph
        kg = KnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

        # Delete existing data
        logger.info("Deleting existing knowledge graph...")
        kg.delete_all()

        # Create constraints
        logger.info("Creating constraints...")
        constraints = [
            ("Episode", "title"),
            ("Person", "name"),
            ("Organization", "name"),
            ("Technology", "name"),
            ("Topic", "name")
        ]
        for label, property in constraints:
            kg.create_constraint(label, property)

        # Build the graph
        kg.build_graph(data)

        # Example queries
        logger.info("Running example queries...")
        
        queries = [
            ("Episodes discussing 'Deep Neural Networks' (top 5)", """
            MATCH (e:Episode)-[:DISCUSSES]->(t:Technology)
            WHERE t.name CONTAINS 'Deep Neural Networks'
            RETURN e.title, e.id
            LIMIT 5
            """),
            ("Most covered topics", """
            MATCH (t:Topic)<-[:COVERS]-(e:Episode)
            RETURN t.name, COUNT(e) as episode_count
            ORDER BY episode_count DESC
            LIMIT 5
            """),
            ("Guest connections through shared organizations", """
            MATCH (g1:Guest)<-[:FEATURES]-(e1:Episode)-[:FEATURES]->(o:Organization)<-[:FEATURES]-(e2:Episode)-[:FEATURES]->(g2:Guest)
            WHERE g1 <> g2
            RETURN g1.name, g2.name, o.name AS shared_organization, COUNT(o) AS connection_strength
            ORDER BY connection_strength DESC
            LIMIT 5
            """),
            ("Most versatile guests", """
            MATCH (g:Guest)<-[:FEATURES]-(e:Episode)-[:COVERS]->(t:Topic)
            WITH g, COUNT(DISTINCT t) as topic_count
            RETURN g.name, topic_count
            ORDER BY topic_count DESC
            LIMIT 5
            """)
        ]

        for description, query in queries:
            result = kg.run_query(query)
            logger.info(f"{description}: {result}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if 'kg' in locals():
            kg.close()

if __name__ == "__main__":
    main()
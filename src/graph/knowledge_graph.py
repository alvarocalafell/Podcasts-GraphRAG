from neo4j import GraphDatabase
import pandas as pd
import logging
from tqdm import tqdm
import ast
from src.utils.config import (
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return list(result)

    def create_constraint(self, label, property):
        create_query = f"CREATE CONSTRAINT {label}_{property}_unique IF NOT EXISTS FOR (n:{label}) REQUIRE n.{property} IS UNIQUE"
        self.run_query(create_query)

    def create_lex_fridman_node(self):
        query = """
        MERGE (l:Person {name: 'Lex Fridman'})
        SET l:Host
        RETURN l
        """
        self.run_query(query)

    def create_episode_node(self, row):
        query = """
        MERGE (e:Episode {title: $title})
        SET e.id = $id
        WITH e
        MATCH (l:Person {name: 'Lex Fridman'})
        MERGE (l)-[:HOSTS]->(e)
        RETURN e
        """
        self.run_query(query, {"id": str(row['id']), "title": row['title']})

    def create_guest_node(self, guest, episode_title):
        query = """
        MERGE (g:Person {name: $guest})
        SET g:Guest
        WITH g
        MATCH (e:Episode {title: $episode_title})
        MERGE (e)-[:FEATURES]->(g)
        """
        self.run_query(query, {"guest": guest, "episode_title": episode_title})

    def create_entity_nodes(self, entities, episode_title):
        query = """
        UNWIND $entities as entity
        MERGE (p:Person {name: entity})
        WITH p
        MATCH (e:Episode {title: $episode_title})
        MERGE (e)-[:MENTIONS]->(p)
        """
        self.run_query(query, {"episode_title": episode_title, "entities": entities})

    def create_organization_nodes(self, organizations, episode_title):
        query = """
        UNWIND $organizations as org
        MERGE (o:Organization {name: org})
        WITH o
        MATCH (e:Episode {title: $episode_title})
        MERGE (e)-[:FEATURES]->(o)
        """
        self.run_query(query, {"episode_title": episode_title, "organizations": organizations})

    def create_technology_nodes(self, technologies, episode_title):
        query = """
        UNWIND $technologies as tech
        MERGE (t:Technology {name: tech})
        WITH t
        MATCH (e:Episode {title: $episode_title})
        MERGE (e)-[:DISCUSSES]->(t)
        """
        self.run_query(query, {"episode_title": episode_title, "technologies": technologies})

    def create_topic_nodes(self, topics, episode_title):
        query = """
        UNWIND $topics as topic
        MERGE (t:Topic {name: topic})
        WITH t
        MATCH (e:Episode {title: $episode_title})
        MERGE (e)-[:COVERS]->(t)
        """
        self.run_query(query, {"episode_title": episode_title, "topics": topics})

    def build_graph(self, data):
        logging.info("Building knowledge graph...")
        self.create_lex_fridman_node()
        for _, row in tqdm(data.iterrows(), total=len(data), desc="Building Graph"):
            self.create_episode_node(row)
            self.create_guest_node(row['guest'], row['title'])
            self.create_entity_nodes(ast.literal_eval(row['people']), row['title'])
            self.create_organization_nodes(ast.literal_eval(row['organizations']), row['title'])
            self.create_technology_nodes(ast.literal_eval(row['technologies']), row['title'])
            self.create_topic_nodes(ast.literal_eval(row['topics']), row['title'])
        logging.info("Knowledge graph built successfully!")

    def query_graph(self, query):
        return self.run_query(query)

    def delete_all(self):
        query = "MATCH (n) DETACH DELETE n"
        self.run_query(query)
        logging.info("All nodes and relationships have been deleted.")

def main():
    # Load processed data
    logging.info("Loading processed data...")
    data = pd.read_csv('data/updated_toy_processed_dataset.csv')
    logging.info(f"Loaded {len(data)} rows of processed data")

    # Initialize knowledge graph
    # Replace these with your actual Neo4j cloud connection details
    uri = NEO4J_URI
    user = NEO4J_USER
    password = NEO4J_PASSWORD
    
    kg = KnowledgeGraph(uri, user, password)

    try:
        # Delete existing data
        logging.info("Deleting existing knowledge graph...")
        kg.delete_all()

        # Create constraints
        logging.info("Creating constraints...")
        kg.create_constraint("Episode", "title")
        kg.create_constraint("Person", "name")
        kg.create_constraint("Organization", "name")
        kg.create_constraint("Technology", "name")
        kg.create_constraint("Topic", "name")

        # Build the graph
        kg.build_graph(data)

        # Example queries
        logging.info("Running example queries...")
        
        # Query 1: Find all episodes discussing 'Deep Neural Networks'
        query1 = """
        MATCH (e:Episode)-[:DISCUSSES]->(t:Technology)
        WHERE t.name CONTAINS 'Deep Neural Networks'
        RETURN e.title, e.id
        LIMIT 5
        """
        result1 = kg.query_graph(query1)
        logging.info(f"Episodes discussing 'Deep Neural Networks' (top 5): {result1}")

        # Query 2: Find the most covered topics
        query2 = """
        MATCH (t:Topic)<-[:COVERS]-(e:Episode)
        RETURN t.name, COUNT(e) as episode_count
        ORDER BY episode_count DESC
        LIMIT 5
        """
        result2 = kg.query_graph(query2)
        logging.info(f"Most covered topics: {result2}")

        # Query 3: Find connections between guests through shared organizations
        query3 = """
        MATCH (g1:Guest)<-[:FEATURES]-(e1:Episode)-[:FEATURES]->(o:Organization)<-[:FEATURES]-(e2:Episode)-[:FEATURES]->(g2:Guest)
        WHERE g1 <> g2
        RETURN g1.name, g2.name, o.name AS shared_organization, COUNT(o) AS connection_strength
        ORDER BY connection_strength DESC
        LIMIT 5
        """
        result3 = kg.query_graph(query3)
        logging.info(f"Guest connections through shared organizations: {result3}")

        # Query 4: Find the most versatile guests (those who cover the most diverse topics)
        query4 = """
        MATCH (g:Guest)<-[:FEATURES]-(e:Episode)-[:COVERS]->(t:Topic)
        WITH g, COUNT(DISTINCT t) as topic_count
        RETURN g.name, topic_count
        ORDER BY topic_count DESC
        LIMIT 5
        """
        result4 = kg.query_graph(query4)
        logging.info(f"Most versatile guests: {result4}")

    finally:
        kg.close()

if __name__ == "__main__":
    main()
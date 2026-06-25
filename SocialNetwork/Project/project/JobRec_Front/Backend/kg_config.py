import os

from py2neo import Graph


def get_graph():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7688")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "TYH041113")
    return Graph(uri, auth=(user, password))

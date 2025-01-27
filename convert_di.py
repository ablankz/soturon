import networkx as nx
import osmnx as ox

def convert_di(g : nx.DiGraph) -> nx.Graph:
    reversed_graph = g.reverse(copy=True)
    return nx.compose(g, reversed_graph)
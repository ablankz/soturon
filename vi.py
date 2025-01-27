import networkx as nx
import matplotlib.pyplot as plt

def graph_draw(g: nx.Graph):
    nx.draw(g, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.show()
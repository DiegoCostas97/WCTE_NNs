import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.utils import to_networkx

def visualize_graph(data, color, to_undirected=True):
    """
    Simple Matplotlib & Networkx tool for visualizing any Graph.
    It needs a graph as G and the graph classes list as color
    """
    G = to_networkx(data, to_undirected=True)

    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="GnBu")
    plt.show()

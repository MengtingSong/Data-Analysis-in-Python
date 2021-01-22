# Evaluate different recommendation functions...
import networkx as nx


def actual_network(G, X):
    clean_graph = nx.Graph(G)
    # get X's neighbours
    X_nbrs = [n for n in nx.all_neighbors(clean_graph, X)]

    return X_nbrs

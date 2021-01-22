# Create graphs from provided txt files...

import networkx as nx


# create graph from old_edges.txt
def old_networks():
    old_networks = nx.MultiGraph()
    old = open('old_edges.txt', 'r')
    for line in old.readlines():
        temp_edge = []
        temp_edge = line.strip('\n').split('\t')
        old_networks.add_edge(temp_edge[0], temp_edge[1])
    old.close()
    return old_networks


# create graph from new_edges.txt
def new_networks():
    new_networks = nx.MultiGraph()
    new = open('new_edges.txt', 'r')
    for line in new.readlines():
        temp_edge = []
        temp_edge = line.strip('\n').split('\t')
        new_networks.add_edge(temp_edge[0], temp_edge[1])
    new.close()
    return new_networks


# Pick authors that formed at least 10 new connections between 2017-2018
def pick_authors(network_graph):
    pick_list = []
    for n, d in nx.degree(network_graph):
        if d >= 10:
            pick_list.append(n)
    return pick_list








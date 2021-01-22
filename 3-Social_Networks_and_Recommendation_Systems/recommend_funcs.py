# Recommendation functions...
import networkx as nx
import math
import numpy as np


def common_friends_number(G, X):
    clean_graph = nx.Graph(G)
    # get X's neighbours
    X_nbrs = [n for n in nx.all_neighbors(clean_graph, X)]

    # get X's neighbours' neighbours
    all_nbrs_of_nbrs = []
    for nbr in X_nbrs:
        all_nbrs_of_nbrs = all_nbrs_of_nbrs + [n for n in nx.all_neighbors(clean_graph, nbr)]

    # remove duplicated common friends and existing friends
    nbrs_of_nbrs = list(set(all_nbrs_of_nbrs))
    nbrs_of_nbrs = [x for x in nbrs_of_nbrs if x not in X_nbrs]

    # sort common friends by number of common friends with X
    recommend_friends_dic = {}
    recommend_friends_list = []
    for nbr_of_nbr in nbrs_of_nbrs:
        nbrs_friends = [n for n in nx.all_neighbors(clean_graph, nbr_of_nbr)]
        recommend_friends_dic[nbr_of_nbr] = len([x for x in nbrs_friends if x in X_nbrs])
    sort_list = sorted(recommend_friends_dic.items(), key=lambda x: x[1], reverse=True)
    for item in sort_list:
        recommend_friends_list.append(item[0])

    return recommend_friends_list


def jaccard_index(G, X):
    clean_graph = nx.Graph(G)
    # get X's neighbours
    X_nbrs = [n for n in nx.all_neighbors(clean_graph, X)]

    # get X's neighbours' neighbours
    all_nbrs_of_nbrs = []
    for nbr in X_nbrs:
        all_nbrs_of_nbrs = all_nbrs_of_nbrs + [n for n in nx.all_neighbors(clean_graph, nbr)]

    # remove duplicated common friends and existing friends
    nbrs_of_nbrs = list(set(all_nbrs_of_nbrs))
    nbrs_of_nbrs = [x for x in nbrs_of_nbrs if x not in X_nbrs]

    # sort common friends by score of common friends with X
    recommend_friends_dic = {}
    recommend_friends_list = []
    for nbr_of_nbr in nbrs_of_nbrs:
        nbrs_friends = [n for n in nx.all_neighbors(clean_graph, nbr_of_nbr)]
        intersection_len = len([x for x in nbrs_friends if x in X_nbrs])
        union_len = len([x for x in list(set(nbrs_friends + X_nbrs))])
        recommend_friends_dic[nbr_of_nbr] = intersection_len / union_len
    sort_list = sorted(recommend_friends_dic.items(), key=lambda x: x[1], reverse=True)
    for item in sort_list:
        recommend_friends_list.append(item[0])

    return recommend_friends_list


def adamic_adar_index(G, X):
    clean_graph = nx.Graph(G)
    # get X's neighbours
    X_nbrs = [n for n in nx.all_neighbors(clean_graph, X)]

    # get X's neighbours' neighbours
    all_nbrs_of_nbrs = []
    for nbr in X_nbrs:
        all_nbrs_of_nbrs = all_nbrs_of_nbrs + [n for n in nx.all_neighbors(clean_graph, nbr)]

    # remove duplicated common friends and existing friends
    nbrs_of_nbrs = list(set(all_nbrs_of_nbrs))
    nbrs_of_nbrs = [x for x in nbrs_of_nbrs if x not in X_nbrs]

    # sort common friends by score of common friends with X
    recommend_friends_dic = {}
    recommend_friends_list = []
    for nbr_of_nbr in nbrs_of_nbrs:
        nbrs_friends = [n for n in nx.all_neighbors(clean_graph, nbr_of_nbr)]
        common_friends = [x for x in nbrs_friends if x in X_nbrs]
        # calculate the common friends z's score
        score = 0
        for z in common_friends:
            score = score + np.reciprocal(math.log(len([n for n in nx.all_neighbors(clean_graph, z)])))
        recommend_friends_dic[nbr_of_nbr] = score
    sort_list = sorted(recommend_friends_dic.items(), key=lambda x: x[1], reverse=True)
    for item in sort_list:
        recommend_friends_list.append(item[0])

    return recommend_friends_list

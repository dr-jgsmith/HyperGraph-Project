#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 09:59:30 2018

@author: justinsmith
"""
import numpy as np
import networkx as nx

"""
Special utility functions for computing with graphs in the NetworkX library
Useful for constructing graphs from various data produced at different intervals
of the Q-Analysis procedure.

"""
def compute_graph_of_inc(incident, row_headers, col_headers):
    edges = []
    count = 0
    for i in incident:
        cnt = 0
        for j in i:
            if j == 1:
                edges.append((row_headers[count].upper(), col_headers[cnt]))
            else:
                pass
            cnt = cnt + 1
        count = count + 1
    return edges


def construct_weighted_graph(qface_matrix, row_headers):
    """
    Takes the shared-face matrix q-1 computed during a q-analysis sequence.
    the ij value of a in A corresponds to the the weighted relation between two simplicies.

    """
    edges = []
    count = 0
    for i in qface_matrix:
        cnt = 0
        for j in i:
            if j > 0:
                edges.append((row_headers[count], row_headers[cnt], j))
            else:
                pass
            cnt = cnt + 1
        count = count + 1
    return edges  


def compute_comp_on_graph(edges):
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    comp = nx.connected_components(G)
    return list(comp)


def construct_bipart_graph(edges):
    bigraph = nx.Graph()
    origin = [i[0] for i in edges]
    destin = [i[1] for i in edges]
    bigraph.add_nodes_from(sorted(set(origin)), bipartite=0)
    bigraph.add_nodes_from(sorted(set(destin)), bipartite=1)
    bigraph.add_edges_from(edges)
    return bigraph


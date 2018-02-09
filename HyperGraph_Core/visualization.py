#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:03:26 2018

@author: justinsmith
"""

"""
Visualization functions for processing graphs, hypergraphs and simplicial complexes.

"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib
import seaborn as sns


def visualize_q_percolation(Q):
    values = [i[1] for i in Q.items()]
    plt.ylabel('Number of Components for Q-Dimension')
    plt.title('Q-Value Percolation')
    plt.plot(values)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()
    
    
def visualize_p_percolation(P):
    values = [i[1] for i in P.items()]
    plt.ylabel('Number of Simplicies for Q-Dimension')
    plt.title('P-Value Percolation')
    plt.plot(values)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()
    
def visualize_q(Q, simplex_set):
    labels = [i for i in simplex_set]
    values = [i for i in Q]
    pos = np.arange(len(labels))
    plt.bar(pos, values, align='center')
    plt.xticks(pos, labels)
    plt.ylabel('Dimensions of Simplex')
    plt.title('Q Structure Vector')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()
    
def visualize_q_slice(Q, simplex_set, theta=1):
    terms = {}
    count = 0
    for i in Q:
        if i > theta:
            terms[simplex_set[count]] = i
        else:
            pass
        count = count + 1 
    values = []
    labels = []
    for i in terms.items():
        values.append(i[1])
        labels.append(i[0])  
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, values, align='center')
    plt.xticks(y_pos, labels)
    plt.ylabel('Number of Shared Vertices')
    plt.title('Top Simplicies')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()
    return terms
    
    
def visualize_eccentricity(simplex_ecc):
    labels = [i[0] for i in simplex_ecc.items()]
    values = [i[1] for i in simplex_ecc.items()]
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, values, align='center')
    plt.xticks(y_pos, labels)
    plt.ylabel('Eccentricity 0 to 1')
    plt.title('Simplex Eccentricity')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()
 

def visualize_retained_ecc(ecc):
    terms = {}
    for i in ecc.items():
        if i[1] > 0:
            terms[i[0]] = i[1]
        else:
            pass
    labels = [i[0] for i in terms.items()]
    values = [i[1] for i in terms.items()]
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, values, align='center')
    plt.xticks(y_pos, labels)
    plt.ylabel('Eccentricity 0 to 1')
    plt.title('Retained Topics By Simplex Eccentricity')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()
    return terms
    
def visualize_simple_graph(I, IT):
    am = (np.dot(I.as_matrix(), IT.as_matrix()) > 0).astype(int)
    #np.fill_diagonal(am, 0)
    G = nx.from_numpy_matrix(am)
    #Draw simplex graph
    pos = nx.shell_layout(G)
    nx.draw(G, pos)
    # show graph
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()
    
    
def visualize_weighted_graph(edges):
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, with_labels=True)
    # show graph
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()
    
    
def visualize_incidence_graph(edges):
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, with_labels=True)
    # show graph
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()
    
    
def visualize_bipart_graph(bigraph):
    l, r = nx.bipartite.sets(bigraph)
    pos = {}
    # Update position for node from each group
    pos.update((node, (1, i)) for i, node in enumerate(l))
    pos.update((node, (2, i)) for i, node in enumerate(r))
    nx.draw(bigraph, pos=pos, with_labels=True)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()
    
def visualize_qmatrix(qmatrix):
    sns.set()
    ax = sns.heatmap(qmatrix)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()
    
def visualize_conjugate(conj):
    sns.set()
    ax = sns.heatmap(conj)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()
    
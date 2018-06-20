import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib
import seaborn as sns

"""
Visualization functions for processing graphs, hypergraphs and simplicial 
complexes. TODOs add options for saving visuals. 
"""


def visualize_q_percolation(Q):
    """
    Takes a python dictionary of variables (keys) and values
    :param Q:
    :return:
    """
    values = Q
    plt.ylabel('Number of Components for Q-Dimension')
    plt.title('Q-Value Percolation')
    plt.plot(values)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()


def visualize_p_percolation(P):
    """
    Takes a python dictionary of variables (keys) and values
    :param P:
    :return:
    """
    values = P
    plt.ylabel('Number of Simplicies for Q-Dimension')
    plt.title('P-Value Percolation')
    plt.plot(values)
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()


def visualize_q(Q, simplex_set):
    """
    :param Q:
    :param simplex_set:
    :return:
    """
    labels = simplex_set
    values = Q
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


def visualize_eccentricity(ecc):
    labels = range(len(ecc))
    values = ecc
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, values, align='center')
    plt.xticks(y_pos, labels)
    plt.ylabel('Eccentricity 0 to 1')
    plt.title('Simplex Eccentricity')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()


def visualize_retained_ecc(ecc):
    labels = range(len(ecc))
    values = ecc
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, values, align='center')
    plt.xticks(y_pos, labels)
    plt.ylabel('Eccentricity 0 to 1')
    plt.title('Retained Topics By Simplex Eccentricity')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()
    return


def visualize_simple_graph(I, IT):
    am = (np.dot(I.as_matrix(), IT.as_matrix()) > 0).astype(int)
    # np.fill_diagonal(am, 0)
    G = nx.from_numpy_matrix(am)
    # Draw simplex graph
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


def visualize_pri_histogram(ranking):
    labels = range(len(ranking))
    values = ranking
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, values, align='center')
    plt.xticks(y_pos, labels)
    plt.ylabel('Preference Ranking Score')
    plt.title('Preference Ranking Index')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()


def visualize_pri_line(ranking, theta):
    labels = range(len(ranking))
    values = ranking
    y_pos = np.arange(len(labels))
    plt.plot(y_pos, values, "s-")
    plt.xticks(y_pos, labels)
    plt.ylabel('Preference Ranking Score')
    plt.title('Preference Ranking Index')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()


def visualize_psi_histogram(psi, theta):
    labels = range(len(psi))
    values = psi
    y_pos = np.arange(len(labels))
    plt.bar(y_pos, values, align='center')
    plt.xticks(y_pos, labels)
    plt.ylabel('Preference Satisfaction Score')
    plt.title('Preference Satisfaction Index')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()


def visualize_psi_line(psi, theta):
    labels = range(len(psi))
    values = psi
    y_pos = np.arange(len(labels))
    plt.plot(y_pos, values, "o-")
    plt.xticks(y_pos, labels)
    plt.ylabel('Preference Satisfaction Score')
    plt.title('Preference Satisfaction Index')
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(10, 8)
    plt.show()

from numba import jit
import numpy as np
from scipy.sparse import csr_matrix, csgraph
import networkx as nx
"""
These functions use the Numba JIT compiler to speed up computational performance on
large data arrays and matrices. These are particularly useful for massive matrix multiplication
problems.

This is a work in progress!!! Changes will be made...

"""


def compute_pattern(traffic_pattern_list, incidence):
    """
    In Q-Analysis it is often the case that we want to analyze the dynamics that occur on a representation/backcloth
    This is computed by multiplying the pattern vector on the incidence matrix of 0s and 1s.
    Here the pattern vector correspond to the vertices.

    convert vector weights into a numpy array

    :param traffic_pattern_list: ordered list of values
    :param incidence: numpy matrix
    :return: numpy matrix
    """
    try:
        vweights = np.array(traffic_pattern_list)
        # convert matrix to a numpy array
        vmatrix = np.array(incidence)
        # multiply the v-weights and numpy array (matrix)
        new_matrix = vweights * vmatrix
        # The new_matrix can now be processed again by passing the matrix to the incidentMatrix method
        return new_matrix
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


def add_cover_pattern(cover_array, incidence):
    """
    This function is used to add a new dimension of relation to a given matrix composed of simplicies and vertices.
    This is accomplished by the addition between two vectors.

    :param cover_array: numpy array
    :param incidence: numpy array
    :return: numpy array
    """
    try:
        cover = np.array(cover_array)
        matrix = np.array(incidence)
        new_matrix = cover + matrix
        return new_matrix
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


@jit
def invert_pattern(pattern_vector):
    """
    Takes a pattern vector in binarized format and inverts the pattern
    :param pattern_vector: numpy array | 0's and 1's
    :return: numpy array
    """
    inverted = np.zeros(len(pattern_vector))
    for i in range(len(pattern_vector)):
        if pattern_vector[i] == 0.0:
            inverted[i] = 1.0
        else:
            inverted[i] = 0.0
    return inverted


@jit
def invert_matrix(matrix):
    inv_matrix = ['x']
    for i in matrix:
        inv = invert_pattern(i)
        inv_matrix.append(inv)
    return np.array(inv_matrix[1:])


@jit
def normalize(matrix):
    normed = ['x']
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[i])):
            val = (matrix[i][j] - min(matrix[:, j])) / (max(matrix[:, j]) - min(matrix[:, j]))
            row.append(val)
        normed.append(row)
    return np.array(normed[1:])


@jit
def compute_incident(value_matrix, theta, slice_type='upper'):
    if slice_type is 'upper':
        data = value_matrix >= theta
    else:
        data = value_matrix <= theta
    return data.astype(int)


def sparse_graph(incidence, theta):
    """
    This function encodes a sparse matrix into a graph representation.
    This function provides a speed up in computation over the numpy matrix methods
    It can be used on both a shared face matrix or raw data input.

    :param incidence: numpy incidence matrix
    :param hyperedge_list: python list | simplicies or nodes
    :param theta: int
    :return: list of tuples
    """
    try:
        sparse = np.nonzero(incidence)
        edges = [(simplex, vertex, incidence[simplex][vertex]) for simplex, vertex in zip(sparse[0], sparse[1]) if incidence[simplex][vertex] >= float(theta)]
        return edges
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


def dowker_relation(sparse_graph):
    """
    This provides a fast approach to computing the shared-face relation between simplicies.
    The function takes a sparse graph and its conjugate as inputs. Returns the dwoker relation + 1.
    To compute the true relation, subtract the -1 from the relation.

    :param sparse_graph: list of tuples
    :param conjugate_graph: list of tuples
    :return: numpy matrix
    """
    try:
        sparseg = compute_graph_matrix_sparse(sparse_graph)
        conjq = sparseg.transpose()
        q_matrix = sparseg.dot(conjq).toarray()
        q_matrix = np.subtract(q_matrix, 1)
        return q_matrix
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


def compute_classes(edges):
    """
    Collect all connected components - Identify equivelence classes
    These are the q-connected components.
    This is central data type for exploring multi-dimensional persistence of Eq Classes
    Takes a list of tuple edges
    :param edges: sparse graph
    :return: list of sets
    """
    try:
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        comp = nx.connected_components(G)
        return  sorted(list(comp))
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


def compute_class_graph(comp_list):
    """
    The ith value in the graph repreents the simplicial complex, will the jth value represents the simplex it is attached to the dimension.
    :param comp_list: a list of sets representing connected simplicies
    :return: graph representation with component indexed by location in the complex set
    """
    try:
        class_graph = [(i, j, 1) for i in range(len(comp_list)) for j in comp_list[i]]
        return class_graph
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


def compute_graph_matrix(sparse_graph):
    """
    Takes the constructed graph of a matrix of simplicial complexes and computes the sparse representation
    :param class_graph: list of tuples
    :return: dense matrix
    """
    try:
        row = np.array([i[0] for i in sparse_graph])
        col = np.array([i[1] for i in sparse_graph])
        data = np.array([i[2] for i in sparse_graph])
        matrix = csr_matrix((data, (row, col)), shape=(max(row)+1, max(col)+1)).toarray()
        return matrix
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


def compute_graph_matrix_sparse(sparse_graph):
    """
    Takes the constructed graph of a matrix of simplicial complexes and computes the sparse representation
    :param sparse_graph: list of tuples
    :return sparse matrix
    """
    try:
        row = np.array([i[0] for i in sparse_graph])
        col = np.array([i[1] for i in sparse_graph])
        data = np.array([i[2] for i in sparse_graph])
        matrix = csr_matrix((data, (row, col)), shape=(max(row) + 1, max(col) + 1))
        return matrix
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


@jit
def compute_paths(sparse_matrix, simplex_index):
    """
    :param sparse_matrix: csr_matrix scipy
    :param simplex_index: vector of simplex labels - integer values representing label index.
    :return: arrays
    """
    seen = np.array([simplex_index])
    fronts = [np.array([simplex_index])]
    for i in fronts:
        tmp = []
        for j in i:
            data = sparse_matrix.getrow(j).nonzero()[1]
            new = np.setdiff1d(data, seen)
            seen = np.union1d(new, seen)
            tmp.extend(new)

        if len(np.unique(tmp)) > 0:
            fronts.append(np.unique(tmp))
        else:
            pass
    return fronts

@jit
def compute_path(sparse_matrix, simplex):
    """
    :param sparse_matrix: csr_matrix scipy
    :param simplex_index: vector of simplex labels - integer values representing label index.
    :return: arrays
    """
    seen = np.array([simplex])
    fronts = [np.array([simplex])]
    for i in fronts:
        tmp = []
        for j in i:
            data = sparse_matrix.getrow(j).nonzero()[1]
            new = np.setdiff1d(data, seen)
            seen = np.union1d(new, seen)
            tmp.extend(new)

        if len(np.unique(tmp)) > 0:
            fronts.append(np.unique(tmp))
        else:
            pass
    return fronts

@jit
def sum_class_matrix(matrix, axis_val):
    """
    :param matrix:
    :param axis_val:
    :return:
    """
    sums = np.sum(matrix, axis=axis_val)
    return sums


@jit
def simple_qanalysis(value_matrix, slicing_list):
    """
    :param value_matrix:
    :param slicing_list:
    :return:
    """
    qgraph = ['x']
    for i in slicing_list:
        incident = compute_incident(value_matrix, i)
        graph = sparse_graph(incident, 1)
        r = dowker_relation(graph)
        graph = sparse_graph(r, 0)
        classes = compute_classes(graph)
        qgraph.append(classes)
    return qgraph[1:]


@jit
def compute_q_structure(qgraph):
    """
    :param qgraph:
    :return:
    """
    qstruct = ['x']
    for i in qgraph:
        qstruct.append(len(i))
    return qstruct[1:]


@jit
def compute_p_structure(qgraph):
    """
    :param qgraph:
    :return:
    """
    pstruct = ['x']
    for j in qgraph:
        tmp = []
        for i in j:
            tmp.append(len(i))
        pstruct.append(sum(tmp))
    return pstruct[1:]


@jit
def simple_ecc(array):
    """
    Compute the eccentricity of a simplex. Produced by computing the Dowker relation
    This takes a 1d array, which is typically the simplex and q-near vertices

    :param array: numpy array
    :return: numpy array
    """
    loc = array.argmax()
    new = np.delete(array, loc)
    qhat = max(array)-1
    qbottom = max(new)-1
    ecc = (qhat - qbottom) / (qbottom + 1)
    return ecc


@jit
def eccentricity(qmatrix):
    """
    takes a q-matrix computed from the dowker relation function
    :param qmatrix: numpy array
    :return: list of eccentricity values
    """
    try:
        # iterate through the matrix to compute the
        eccs = [simple_ecc(i) for i in qmatrix]
        return eccs
    except MemoryError:
        print('Memory Error')
        pass
    except RuntimeError:
        print('Runtime Error')
        pass
    except TypeError:
        print('Type Error')
        pass
    except NameError:
        print('Name Error')
        pass


@jit
def chin_ecc(comps, hyper_edges):
    """
    :param comps:
    :param hyper_edges:
    :return:
    """
    strct = len(hyper_edges) - 1
    ecc = ['x']
    for i in hyper_edges:
        tmp = []
        cnt = 0
        for j in comps:
            for k in j:
                if i in k:
                    e = cnt / len(k)
                    tmp.append(e)
                else:
                    pass
            cnt = cnt + 1
        ecc.append(sum(tmp))
    qmax = (1/2 * strct) * (strct + 1)
    ecc = np.divide(ecc[1:], qmax)
    return ecc


@jit
def compute_psi(value_matrix, weights, slice_list):
    """
    :param value_matrix:
    :param weights:
    :param slice_list:
    :return:
    """
    psi = ['x']
    psimax = ['x']
    for i in slice_list:
        incident = compute_incident(value_matrix, i)
        data = compute_pattern(weights, incident)
        repr = np.zeros(len(data))
        for i in range(len(data)):
            repr[i] = sum(data[i])
        psi.append(repr)
        psimax.append(sum(weights))

    scores = np.array(psi[1:]).sum(axis=0)
    maxscore = np.array(psimax[1:]).sum(axis=0)
    psi = np.divide(scores, maxscore)
    return psi


@jit
def compute_pci(value_matrix, slice_list):
    """
    :param value_matrix:
    :param slice_list:
    :return:
    """
    pci = ['x']
    pcimax = ['x']
    for i in slice_list:
        incident = compute_incident(value_matrix, i)
        g = sparse_graph(incident, 1)
        Q = dowker_relation(g)
        strct = Q.diagonal()
        pmax = len(strct) - 1
        tmp = np.zeros(len(strct))
        tmax = np.zeros(len(strct))
        for j in range(len(Q)):
            # q = max([Q[j][i] for i in range(len(Q[j])) if i is not j])
            q = []
            for k in range(len(Q[j])):
                if k is not j:
                    q.append(Q[j][k])

            if strct[j] < 0:
                tmp[j] = 0
                tmax[j] = pmax
            else:
                val = strct[j] - max(q)
                tmp[j] = val
                tmax[j] = pmax
        pci.append(tmp)
        pcimax.append(tmax)
    pci = np.array(pci[1:]).sum(axis=0)
    pcimax = np.array(pcimax[1:]).sum(axis=0)
    pci = np.divide(pci, pcimax)
    return pci


@jit
def compute_pdi(value_matrix, slice_list):
    """
    :param value_matrix:
    :param slice_list:
    :return:
    """
    pdi = ['x']
    pdimax = ['x']
    for i in slice_list:
        incident = compute_incident(value_matrix, i)
        complement = invert_matrix(incident)
        g = sparse_graph(complement, 1)
        Q = dowker_relation(g)
        strct = Q.diagonal()
        pmax = len(strct) - 1
        tmp = np.zeros(len(strct))
        tmax = np.zeros(len(strct))
        for j in range(len(Q)):
            q = []
            for k in range(len(Q[j])):
                if k is not j:
                    q.append(Q[j][k])

            if strct[j] < 0:
                tmp[j] = 0
                tmax[j] = pmax
            else:
                val = strct[j] - max(q)
                tmp[j] = val
                tmax[j] = pmax
        pdi.append(tmp)
        pdimax.append(tmax)
    pdi = np.array(pdi[1:]).sum(axis=0)
    pdimax = np.array(pdimax[1:]).sum(axis=0)
    pdi = np.divide(pdi, pdimax)
    return pdi


def mcqa_ranking_I(psi, pci):
    """
    :param psi:
    :param pci:
    :return:
    """
    pri_psi = np.subtract(1, psi)
    pri_pci = np.subtract(1, pci)
    pri = np.add(pri_psi, pri_pci)
    return pri


def mcqa_ranking_II(psi, pci, pdi):
    '''
    :param psi:
    :param pci:
    :param pdi:
    :return:
    '''
    pri_psi = np.subtract(1, psi)
    pri_pci = np.subtract(1, pci)
    pri_partial = np.add(pri_psi, pri_pci)
    pri = np.add(pri_partial, pdi)
    return pri


def compute_mcqa(value_matrix, criteria, slicing_list):
    """
    :param value_matrix:
    :param criteria:
    :param slicing_list:
    :return:
    """
    norm = normalize(value_matrix)
    psin = compute_psi(norm, criteria, slicing_list)
    pcin = compute_pci(norm, slicing_list)
    pdin = compute_pdi(norm, slicing_list)

    mcq = mcqa_ranking_I(psin, pcin)
    mcq2 = mcqa_ranking_II(psin, pcin, pdin)

    return mcq, mcq2


# Compute system complexity.
# This is only one of many possible complexity measures and is provided by John Casti.
def compute_complexity(q_percolation):
    """
    :param q_percolation:
    :return:
    """
    strct = []
    vect = []
    for i in q_percolation.items():
        x = i[0] + 1
        y = x * i[1]
        strct.append(y)
        vect.append(i[1])
    z = sum(strct)
    complexity = 2 * (z / ((max(vect) + 1) * (max(vect) + 2)))
    return complexity


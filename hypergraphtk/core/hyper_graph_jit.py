from numba import jit
import numpy as np
import networkx as nx
from collections import defaultdict
"""
These functions use the Numba JIT compiler to speed up computational performance on
large data arrays and matrices. These are particularly useful for massive matrix multiplication
problems.

This is a work in progress!!! Changes will be made...

"""
# Optional: Step 0
@jit
def vector_weights(wght_list, matrix):
    """
    :param wght_list: ordered list of values
    :param matrix: numpy
    :return: numpy
    """
    # This method combines a vector of weights and a matrix (pre-normalized)
    # These weights can be defined on specific simplicies as a dictionary of simplicies and their weighted value
    # However, weights can be generated based on a rule or probability distribution.
    # get values for the weights
    # convert vector weights into a numpy array
    vweights = np.array(wght_list)
    # convert matrix to a numpy array
    vmatrix = np.array(matrix)
    # multiply the v-weights and numpy array (matrix)
    new_matrix = vweights * vmatrix
    return new_matrix


# Optional: Step 0
@jit
def normPositive(matrix):
    """
    :param matrix: numpy
    :return: 
    """
    # Normalize a matrix given a raw set of values. This produces a positive value.
    # Optional transformation of input matrix given a set of raw values. This function is
    # typically used in conjunction with a vector of weights.
    normed_matrix = []
    for i in matrix:
        x = max(i)
        y = min(i)
        row = []
        for j in i:
            a = (j - y) / (x - y)
            row.append(a)
        normed_matrix.append(row)
    normed_matrix = np.vstack(normed_matrix)
    return normed_matrix


# Optional: Step 0
@jit
def normNeg(matrix):
    """
    :param matrix: numpy
    :return: 
    """
    # Normalize a matrix given a raw set of values. This produces a negative value.
    # Optional transformation of input matrix given a set of raw values. This function is
    # typically used in conjunction with a vector of weights. This is used when negative value
    # is desirable as a criteria.
    normed_matrix = []
    for i in matrix:
        x = max(i)
        y = min(i)
        row = []
        for j in i:
            a = (x - j) / (x - y)
            row.append(a)
        normed_matrix.append(row)
    normed_matrix = np.vstack(normed_matrix)
    return normed_matrix


# Optional: Step 0
# Note this step can be repeated
@jit
def computePattern(traffic_pattern_list, incidence):
    """
    :param traffic_pattern_list: ordered list of values
    :param incidence: numpy matrix
    :return: numpy matrix
    """
    # In Q-Analysis it is often the case that we want to analyze the dynamics that occur on a representation/backcloth
    # This is computed by multiplying the pattern vector on the incidence matrix of 0s and 1s.
    # Here the pattern vector correspond to the vertices.
    #wght = [i[1] for i in traffic_pattern.items()]
    # convert vector weights into a numpy array
    vweights = np.array(traffic_pattern_list)
    # convert matrix to a numpy array
    vmatrix = np.array(incidence)
    # multiply the v-weights and numpy array (matrix)
    new_matrix = vweights * vmatrix
    # The new_matrix can now be processed again by passing the matrix to the incidentMatrix method
    return new_matrix


# Step 1: Construct Incidence Matrix from a given matrix and threshold or slicing parameter.
@jit
def incidentMatrix(matrix, theta, norm=False, less_than=False):
    """
    :param matrix: numpy matrix
    :param theta: float scalar value
    :param norm: bool
    :param less_than: bool
    :return: numpy matrix
    """
    # This function provides a basic method for describing a relation between two sets that
    # have been computed as a MxN matrix of values that map to simplicies (rows) and vertices (columns).
    # The theta value represents a threshold parameter for defining the partition of the matrix into
    # 0's and 1's.
    if norm is True:
        new_matrix = normPositive(matrix)
    else:
        new_matrix = matrix

    incident = [np.ones(len(new_matrix[0]))]
    if less_than is True:
        for j in new_matrix:
            k = np.piecewise(j, [j > theta, j <= theta], [0, 1])
            incident.append(k)
    else:
        for j in new_matrix:
            k = np.piecewise(j, [j > theta, j <= theta], [1, 0])
            incident.append(k)

    incident = np.vstack(incident[1:])
    return incident

 # Step 2: Compute the conjugate or transpose of the retained inicidence matrix
@jit
def computeConjugate(incident):
    """
    :param incident: numpy matrix
    :return: numpy matrix
    """
    # Compute the conjugate of the incidence matrix
    incident_transpose = incident.transpose()
    return incident_transpose

# Step 3: Multiply the incidence by its transpose - BxB^T
# This produces the shared-faced-matrix of the incidence and its transpose
@jit
def computeQFace(incident, incident_transpose):
    # Multiply the incidence matrix by its transpose
    shared_face_matrix = np.matmul(incident, incident_transpose)
    return shared_face_matrix

# Step 4: Compute the Q Matrix by subtracting 1 from each value in the matrix
@jit
def computeQMatrix(shared_face_matrix):
    # compute the Q-Matrix as the shared face matrix minus E
    E = np.ones(shared_face_matrix.shape).astype(int)
    qmatrix = np.subtract(shared_face_matrix, E)
    return qmatrix

# Step 5: Extract the Q Structure Vector from the Q Matrix
@jit
def computeQStruct(qmatrix):
    # Extract the first Q Structure vector
    matrix = np.array(qmatrix)
    q_vector = matrix.diagonal()
    return q_vector

@jit
def computeSparseQ(incident):
    incident_transpose = incident.transpose()
    shared_face_matrix = np.matmul(incident, incident_transpose)
    E = np.ones(shared_face_matrix.shape).astype(int)
    qmatrix = np.subtract(shared_face_matrix, E)
    return qmatrix


# Step 6: This is step converts the qmatrix into a weighted graph
# The weighted graph can then be used to compute qnear simplicies (shortest path and all connected components)
def construct_weighted_graph(shared_face, hyperedge_set, vertex_set, conjugate=False):
    """
    Takes the shared-face matrix q-1 computed during a q-analysis sequence.
    the ij value of a in A corresponds to the the weighted relation between two simplicies.
    """
    if conjugate is False:
        hyperedges = hyperedge_set
    else:
        hyperedges = vertex_set

    G = nx.from_numpy_matrix(shared_face)
    print("Extracting weighted edges")
    edges = [(hyperedges[i[0]], hyperedges[i[1]], shared_face[i[0]][i[1]]) for i in nx.edges(G)]
    return edges


# Step 7. Use the weighted graph to compute homotopy equivalence classes.
# These are the q-connected components.
# This is central data type for exploring multi-dimensional persistence of Eq Classes
def computeEqClasses(edges):
    # Collect all connected components
    # Identify equivelence classes
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    comp = nx.connected_components(G)
    return list(comp)


# Step 8. Compute the q-near simplicies for constructing the Transmission Front paths for each simplex.
def computeQNear(edges):
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    qnear = dict(nx.all_pairs_shortest_path(G))
    return qnear


# Step 9. Compute the transmission fronts.
# This is important and interesting property of the q-analysis method
# Transmission fronts can be used to compute flows given the shared dimensions between simplicies in the path.
def computeFronts(qnear):
    t_fronts = {}
    for i in qnear.items():
        front_list = []
        fronts = defaultdict(list)
        for j in i[1].items():
            dim = len(j[1]) - 1
            fronts[dim].append(j[0])
        [front_list.append(k[1]) for k in fronts.items()]
        t_fronts[i[0]] = front_list
    return t_fronts


# The next set of methods are used for computing diagnostic measures.
# Compute the P-Structure vector of the complex.
# The P-structure vector is used to measure the number of simplicies for a given q-dimension
# This method takes a set of Q-Structure vectors that are computed for each q-dimension
def computeP(qstruct_set):
    pstruct = {}
    dim = 0
    for i in qstruct_set:
        val = 0
        for j in i:
            val = len(j) + val
        pstruct[dim] = val
        dim = dim + 1
    return pstruct

# The Q-Structure vector is used to compute the number of s-homotopy equivalence classes for each q-dimension.
def computeQ(qstruct_set):
    qstruct = {}
    dim = 0
    for i in qstruct_set:
        qstruct[dim] = len(i)
        dim = dim + 1
    return qstruct


# Compute the eccentricity of each simplex in the complex.
# This measures how integrated a simplex is in the complex.
# Note: it would be interesting to compute a similar diagnostic for simplex persistence at each q-dim.
def computeEcc(EqClasses, qstruct, hyperedge_set, vertex_set, conjugate=False):
    if conjugate is False:
        hyperedges = hyperedge_set
    else:
        hyperedges = vertex_set

    # eccI = 2(sum(q_dim/num_simps))/(q_dim*(q_dim+1))
    eccentricity = {}
    for simplex in hyperedges:
        simplex_dim = []
        dim = 0
        for i in EqClasses:
            for j in i:
                if simplex in j:
                    val = dim / len(j)
                    simplex_dim.append(val)
                else:
                    pass
                dim = dim + 1
        # The ecc algorithm is based on the equation provided by Chin et al. 1991
        ecc = sum(simplex_dim) / ((1 / 2 * float(max(qstruct))) * float((max(qstruct) + 1)))
        eccentricity[simplex] = ecc
    return eccentricity


# Compute system complexity.
# This is only one of many possible complexity measures and is provided by John Casti.
def computeComplexity(q_percolation):
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



class hypergraph_jit:
    def __init__(self, hyperedge_set, vertex_set):

        self.hyperedge_set = hyperedge_set
        self.vertex_set = vertex_set

    # The computeSimpleQ method provides simplified version of the computeQanalysis method
    def computeSimpleQ(self, matrix, theta, norm_matrix=False, conjugate=False):
        try:
            incident = incidentMatrix(matrix, theta, norm=norm_matrix, less_than=False)

            if conjugate is False:
                pass
            else:
                incident = computeConjugate(incident)

            print(incident)

            #qmatrix = computeSparseQ(incident)

            incident_transpose = computeConjugate(incident)
            # Matrix multiplication is resource intensive
            shared_face_matrix = computeQFace(incident, incident_transpose)
            print(shared_face_matrix)

            qmatrix = computeQMatrix(shared_face_matrix)
            print("Q Matrix :", qmatrix)

            strct = computeQStruct(qmatrix)
            print("Structure vector computed: ", strct)
            edges = construct_weighted_graph(shared_face_matrix, self.hyperedge_set, self.vertex_set,  conjugate=conjugate)
            print("Edges computed")
            # simplicial complex  methods
            qchains = computeEqClasses(edges)
            print('Classes computed. Num of connected components: ', len(qchains))
            # qnear = self.computeQNear(wedges)
            # print('Neighbors computed ')
            # qfronts = self.computeFronts(qnear)
            return qmatrix, qchains, strct
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


    # This system provides the individual computational methods for performing multi-level systems analysis.
    # However, these methods are typically used together in a sequence, and required to perform a full q-analysis.
    # The computeQanalysis automates the q-analysis of the multi-dimensional relations
    def computeQanalysis(self, matrix, theta, norm_matrix=False, conjugate=False):
        Qchains = []
        Qnear = []
        Qfronts = []

        incident = incidentMatrix(matrix, theta, norm=norm_matrix, less_than=False)

        if conjugate is False:
            pass
        else:
            incident = computeConjugate(incident)

        print(incident)

        incident_transpose = computeConjugate(incident)
        # Matrix multiplication is resource intensive
        shared_face_matrix = computeQFace(incident, incident_transpose)
        print(shared_face_matrix)

        qmatrix = computeQMatrix(shared_face_matrix)
        strct = computeQStruct(qmatrix)

        count = 1
        while count <= max(strct):
            shared_face_matrix = computeQMatrix(shared_face_matrix)
            # graph methods
            edges = construct_weighted_graph(shared_face_matrix, self.hyperedge_set, self.vertex_set, conjugate=conjugate)
            print("Edges computed")
            # simplicial complex  methods
            qchains = computeEqClasses(edges)
            print('Classes computed. Num of connected components: ', len(qchains))

            # simplicial complex  methods
            qnear = computeQNear(edges)
            qfronts = computeFronts(qnear)

            if len(edges) < 1:
                pass
            else:
                Qchains.append(qchains)
                Qnear.append(qnear)
                Qfronts.append(qfronts)

            count = count + 1

        q_percolation = computeQ(Qchains)
        ##Compute P diagnostic
        p_percolation = computeP(Qchains)
        # Compute eccentricity for each simplex/hyper-edge
        eccentricity = computeEcc(Qchains, strct, conjugate=conjugate)
        # Compute complexity of the system
        complexity = computeComplexity(q_percolation)
        return qmatrix, strct, Qchains, Qnear, Qfronts, q_percolation, p_percolation, eccentricity, complexity


    # This method is used primarily to compute Preference Discordance Index
    # The method computes the complement of the original incidence matrix
    # Used to conduct an MCQA optimization
    def computeQcompliment(incident):
        compliment = []
        for i in incident:
            row = []
            for j in i:
                if j == 0:
                    row.append(1)
                elif j == 1:
                    row.append(0)
                elif j == -1:
                    row.append(0)

            compliment.append(row)

        incident_comp = np.array(compliment)
        incident_transpose = computeConjugate(incident_comp)
        shared_face_matrix = computeQFace(incident_comp, incident_transpose)
        qmatrix = computeQMatrix(shared_face_matrix)
        strct = computeQStruct(qmatrix)


        return incident_comp, qmatrix, strct
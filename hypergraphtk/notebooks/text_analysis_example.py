from hypergraphtk.core.hyper_graph import *
from hypergraphtk.visual.visualization import *

cmatrix = [[5, 5, 2, 1, 3, 0],
           [7, 6, 0, 3, 0, 2],
           [2, 2, 1, 0, 1, 0],
           [8, 9, 1, 1, 5, 0],
           [5, 4, 6, 1, 3, 4]]

cut_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

normed = normalize(np.array(cmatrix))
incident = compute_incident(normed, 0.6)
print(incident)
sg = sparse_graph(incident, 1)
m = dowker_relation(sg)
print(m)

qgraphs = simple_qanalysis(normed, cut_points)
print(qgraphs)

q = compute_q_structure(qgraphs)
print(q)
visualize_q_percolation(q)

p = compute_p_structure(qgraphs)
print(p)
visualize_p_percolation(p)

ecc = chin_ecc(qgraphs, range(len(cmatrix)))
print(ecc)
visualize_eccentricity(ecc)
from hypergraphtk.core.hyper_graph import *
from hypergraphtk.visual.visualization import *


A = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

B = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

matrix = [[0, 0, 7, 0, 0, 6, 3, 2],
          [5, 4, 7, 0, 0, 0, 4, 0],
          [0, 0, 0, 6, 6, 4, 3, 5],
          [0, 3, 7, 7, 3, 5, 5, 4],
          [0, 6, 6, 5, 4, 0, 0, 0],
          [0, 0, 0, 2, 4, 0, 0, 7],
          [0, 2, 2, 5, 0, 0, 0, 6],
          [0, 3, 3, 4, 4, 0, 0, 0],
          [0, 0, 7, 0, 0, 0, 0, 0],
          [2, 7, 2, 2, 0, 0, 0, 6],
          [0, 0, 0, 0, 4, 0, 0, 0],
          [3, 6, 6, 0, 0, 0, 0, 0]]


weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
cut_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

normed = normalize(np.array(matrix))
qgraphs = simple_qanalysis(normed, cut_points)
print(qgraphs)

q = compute_q_structure(qgraphs)
print(q)
visualize_q_percolation(q)

p = compute_p_structure(qgraphs)
print(p)
visualize_p_percolation(p)

ecc = chin_ecc(qgraphs, range(len(matrix)))
print(ecc)
visualize_eccentricity(ecc)



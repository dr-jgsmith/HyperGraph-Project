from hypergraphtk.core.hyper_graph import *
from hypergraphtk.visual.visualization import *

# Example of a multi-criteria decision analysis with MCQA I & II algorithim

# A - set represent a list of decision alternatives
A = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

# B - set represent a list of criteria for ranking decision alternaitves
B = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

# matrix represents the criteria values (columns) for the decision alternatives (rows)
matrix = [[0, 1, 7, 0, 1, 6, 3, 2],
          [5, 4, 7, 0, 0, 1, 4, 1],
          [1, 1, 1, 6, 6, 4, 3, 5],
          [1, 3, 7, 7, 3, 5, 5, 4],
          [1, 6, 6, 5, 4, 0, 0, 0],
          [1, 1, 1, 2, 4, 0, 0, 7],
          [0, 2, 2, 5, 0, 0, 0, 6],
          [0, 3, 3, 4, 4, 0, 0, 0],
          [0, 0, 7, 1, 0, 0, 1, 1],
          [2, 7, 2, 2, 0, 0, 1, 6],
          [1, 1, 1, 0, 4, 0, 0, 0],
          [3, 6, 6, 0, 1, 0, 0, 1]]

# weights can be defined to specify the relative importance of a criteria.
# in this case the weights are equally important
weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# cut points are the threshold parameters for conducting a Q-Analysis
# Each cut-point can be used to represent a scale such as low, medium or high
cut_points = [0.2, 0.4, 0.6, 0.8]

norm = normalize(np.array(matrix))
print(norm)

mqa_i = compute_mcqa_i(np.array(matrix), weights, cut_points)
print(mqa_i)
visualize_pri_histogram(mqa_i)

mqa_ii = compute_mcqa_ii(np.array(matrix), weights, cut_points)
print(mqa_ii)
visualize_pri_histogram(mqa_ii)

mqa_iii = compute_mcqa_iii(np.array(matrix), weights, cut_points)
print(mqa_iii)
visualize_pri_histogram(mqa_iii)


qgraphs = simple_qanalysis(norm, cut_points)
print(qgraphs)

q = compute_q_structure(qgraphs)
print(q)
visualize_q_percolation(q)

p = compute_p_structure(qgraphs)
print(p)
visualize_p_percolation(p)

p = compute_p_structure(qgraphs)
print(p)
visualize_p_percolation(p)

ecc = chin_ecc(qgraphs, range(len(matrix)))
print(ecc)
visualize_eccentricity(ecc)

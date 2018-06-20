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

mqa = compute_mcqa(np.array(matrix), weights, cut_points)
print(mqa)

visualize_pri_histogram(mqa[0])
visualize_pri_histogram(mqa[1])

sg = sparse_graph(norm, 0.4)
sgm = compute_graph_matrix_sparse(sg)
print(sgm)

for i in range(len(A)):
    path = compute_paths(sgm, i)
    print(path)

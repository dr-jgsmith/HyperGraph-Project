from hypergraphtk.core.hyper_graph import *


A = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

B = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

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

weights = dict(a=1.1, b=.97, c=0.4, d=2, e=2.2, f=1.5, g=0.18, h=0.34)
data = hypergraph(A, B)
weighted = data.vector_weights(weights, matrix)
print(weighted)

normed = data.normPositive(matrix=weighted)
print(normed)

q = data.computeQanalysis(normed, theta=0.2, norm_matrix=True, conjugate=False)
print(data.incident)
print(q[0])
print(q[1])

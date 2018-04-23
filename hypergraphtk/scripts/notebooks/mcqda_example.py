from hypergraphtk.core.mcqda import *
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
weights = dict(a=1.0, b=1.0, c=1.0, d=1.0, e=1.0, f=1.0, g=1.0, h=1.0)

# cut points are the threshold parameters for conducting a Q-Analysis
# Each cut-point can be used to represent a scale such as low, medium or high
cut_points = [0.2, 0.4, 0.6, 0.8]

data = mcqda(A, B)
data.set_weights(weights)
data.normMatrix(matrix)
print(data.normed)

# compute ranking for the set of alternatives
x = data.process_mcqda_scales(cut_points)
print(x)

# visualize the results of the ranking index for each cut point or scale
# shows ranking for each scale
for i in x.items():
    for j in i[1].items():
        if j[0] == 'rankI':
            visualize_pri_line(j[1], i[0])
        else:
            pass
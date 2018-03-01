#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:38:04 2018

@author: justinsmith
"""
from q_analysis import *
from hypergraph_kv import HyperGraphDB

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


db = HyperGraphDB('tmp.db')
db.update_from_matrix(A, B, matrix)
matrix2 = db.construct_matrix()

qchains = computeNormalizedQ(matrix2, A, B, label=None, visualize=True, theta=0.2)
#qchains = computeNormalizedConj(matrix2, A, B, label=None, visualize=None, theta=0.2)
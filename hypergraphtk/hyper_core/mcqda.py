#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:22:15 2018

@author: justinsmith

This set of test functions are used for multt-criteria optimization and analysis.
These functions still need work and will eventually be integrated with hypergraph.
"""
import numpy as np


def define_weights(vector_dict):
    weighted_vector = []
    for i in vector_dict.items():
        weighted_vector.append(i[1])
    return weighted_vector
    

def computePSI(q_simplex, weighted_vector):
    matrix = []
    for i in q_simplex:
        tmp = []
        cnt = 0
        for j in i:
            x = weighted_vector[cnt]
            d = x*j
            tmp.append(d)
            cnt = cnt + 1
        matrix.append(tmp)
    psi = np.array(matrix)
    return psi


def computePSIN(psi):
    psin = []
    for i in psi:
        x = max(i)
        row = []
        for j in i:
            if j == 0:
                p = 0
            else:
                if x == 0:
                    p = 0
                else:
                    p = j/x
            row.append(p)
        psin.append(row)
    psina = np.array(psin)
    return psina


#def computePCI():
    
    
#def computePCIN(pci):
    


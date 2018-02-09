#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:22:42 2018

@author: justinsmith

HyperGraph Core 

This file contains the core q-analysis function set. 
These functions include HyperGraph -  Q-Analysis Function Set. 
The function follow in a linear fashion, where each function represents a set 
in the process of q-analysis. The first two functions are however, optional. 

"""
import numpy as np
import networkx as nx
from collections import defaultdict

#Optional: Step 0
def normPositive(matrix):
    #Normalize a matrix given a raw set of values. This produces a positive value.
    #Optional transformation of input matrix given a set of raw values. This function is 
    #typically used in conjunction with a vector of weights.
    normalized_matrix = []
    for i in matrix:
        x = max(i)
        y = min(i)
        row = []
        for j in i:
            a = (j-y)/(x-y)
            row.append(a)
        normalized_matrix.append(row)      
    return normalized_matrix

#Optional: Step 0
def normNeg(matrix):
    #Normalize a matrix given a raw set of values. This produces a negative value.
    #Optional transformation of input matrix given a set of raw values. This function is 
    #typically used in conjunction with a vector of weights. This is used when negative value
    #is desirable as a critera.
    normalized_matrix = []
    for i in matrix:
        x = max(i)
        y = min(i)
        row = []
        for j in i:
            a = (x-j)/(x-y)
            row.append(a)
        normalized_matrix.append(row)      
    return normalized_matrix


#Step 1: Construct Incidence Matrix from a given matrix and threshold or slicing parameter.
def incidentMatrix(matrix, theta=1):
    #This function provides a basic method for describing a relation between two sets that
    #have been computed as a MxN matrix of values that map to simplicies (rows) and vertices (columns).
    #The theta value represents a threshold parameter for defining the partition of the matrix into
    #0's and 1's.
    incident = np.zeros((len(matrix), len(matrix[0]))).astype(int)
    count = 0
    #Iterate through the rows of the matrix
    for i in matrix:
        cnt = 0
        #Iterate through the columns of the matrix
        for j in i:
            if j >= theta:
                incident[count][cnt] = 1
            else:
                incident[count][cnt] = 0
            cnt = cnt+1
        count = count+1
    return np.array(incident)


#Step 2: Compute the conjugate or tranpose of the retained inicidence matrix
def computeConjugate(incident):
    #Compute the conjugate of the incidence matrix
    incident_transpose = incident.transpose()
    return incident_transpose


#Step 3: Multiply the incidence by its transpose - BxB^T
def computeQFace(incident, incident_transpose):
    #Multiply the incidence matrix by its transpose
    shared_face = incident.dot(incident_transpose)
    return shared_face


#Step 4:
def computeQMatrix(shared_face):
    #compute the Q-Matrix as the shared face matrix minus E
    E = np.ones(shared_face.shape).astype(int)
    qmatrix = np.subtract(shared_face, E)
    return qmatrix


#Step 5:
def computeQStruct(qmatrix):
    #Extract the first Q Structure vector
    matrix = np.array(qmatrix)
    q_vector = matrix.diagonal()
    return q_vector


#The next set of functions conpute additional attributes relevant to Q-Analysis
#
def computeEqClasses(edges):
    #Collect all connected compoenents 
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    comp = nx.connected_components(G)
    return list(comp)

def computeQNear(edges):
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    qnear = dict(nx.all_pairs_shortest_path(G))
    return qnear

def computeFronts(qnear):
    F = {}
    for i in qnear.items():
        front_list = []
        fronts = defaultdict(list)
        for j in i[1].items():
            dim = len(j[1]) - 1
            fronts[dim].append(j[0])
        [front_list.append(k[1]) for k in fronts.items()]
        F[i[0]] = front_list
    return F

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:22:42 2018

@author: justinsmith


This is a second or perhaps even third implementation of the Q-Analysis
System. 


This implementation focuses on leveraging the NetworkX Python Lib to construct the 
hypernetworks that will serve in the analysis.

"""
import numpy as np
import networkx as nx
from collections import defaultdict



def incidentMatrix(matrix, theta=1):
    """
    This function provides a basic method for describing a relation between two sets that
    have been computed as a MxN matrix of values that map to simplicies (rows) and vertices (columns).
    
    The theta value represents a threshold parameter for defining the partition of the matrix into
    0's and 1's.
    """
    incident = np.zeros((len(matrix), len(matrix[0]))).astype(int)
    count = 0
    for i in matrix:
        cnt = 0
        for j in i:
            if j >= theta:
                incident[count][cnt] = 1
            else:
                incident[count][cnt] = 0
            cnt = cnt+1
        count = count+1
    return np.array(incident)


def normPositive(C):
    matrix = []
    for i in C:
        x = max(i)
        y = min(i)
        row = []
        for j in i:
            a = (j-y)/(x-y)
            row.append(a)
        matrix.append(row)      
    return matrix

def normNeg(C):
    matrix = []
    for i in C:
        x = max(i)
        y = min(i)
        row = []
        for j in i:
            a = (x-j)/(x-y)
            row.append(a)
        matrix.append(row)      
    return matrix

def computeConjugate(incident):
    incident_transpose = incident.transpose()
    return incident_transpose

def computeQFace(incident, incident_transpose):
    shared_face = incident.dot(incident_transpose)
    return shared_face

def computeQMatrix(shared_face):
    E = np.ones(shared_face.shape).astype(int)
    qmatrix = np.subtract(shared_face, E)
    return qmatrix

def computeQStruct(qmatrix):
    matrix = np.array(qmatrix)
    q_vector = matrix.diagonal()
    return q_vector

def computeEqClasses(edges):
    #np.fill_diagonal(am, 0)
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

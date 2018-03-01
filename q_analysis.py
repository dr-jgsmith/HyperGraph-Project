#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:04:26 2018

@author: justinsmith



The following function represents the main app function for computing system properties

There are different functions to compute the qanalysis.
These include computing q over a singele threshold or a uniform slice.


""" 

import numpy as np
import networkx as nx
from hyper_graph import *
from graph_utils import *
from diagnostic import *
from visualization import *



   
def computeQAnalysis(matrix, row_headers, col_headers, label, visualize=None, theta=1):
    '''
    For now we have to run an inital scan of the data to discover the structure in order
    to compute a complete structural analysis of the data. This could be shortfall of the algortithm.
    '''
    simplicies = row_headers
    verticies = col_headers
    #Return variables
    Qchains = []
    Qnear = []
    Fronts = []
    #compute inital structure vector
    incident = incidentMatrix(matrix, theta=theta)
    conj = computeConjugate(incident)
    qface = computeQFace(incident, conj)
    qmatrix = computeQMatrix(qface)
    #structure vector for computing q-analysis
    structure = computeQStruct(qmatrix)
    cnt = 1
    #Iterate through the complex given the n-dimensional structure
    while cnt <= max(structure):
        #matrix constructor methods
        qface = computeQMatrix(qface)
        #graph methods
        wedges = construct_weighted_graph(qface, simplicies)
        #simplicial complex  methods
        qchains = computeEqClasses(wedges)
        qnear = computeQNear(wedges)
        qfronts = computeFronts(qnear)
        #Retain these attributes for other computing features
        if len(wedges) < 1:
            pass
        else:
            Qchains.append(qchains)
            Qnear.append(qnear)
            Fronts.append(qfronts)
        #visualize methods
        if len(wedges) < 1:
            pass
        else:
            if visualize == True:
                dim = cnt - 1
                print('Q Dimension: ', dim)
                print(label)
                visualize_weighted_graph(wedges)
                print('Q Dimension: ', dim)
                print(label)
                visualize_qmatrix(qmatrix)
        cnt = cnt + 1
    #Compute Q diagnostics
    Q = computeQ(Qchains)
    ##Compute P diagnostic
    P = computeP(Qchains)
    #Compute eccentricity for each simplicie/hyper-edge
    ecc =  Ecc(Qchains, row_headers, structure)
    #Compute complexity of the system
    complexity_val = complexity(Q) 
    #Print results to command line for some simple diagnostic clues.
    #Show visuals for computed diagnostics
    if visualize == True:
        #Visualize Eccentricity of Simplicies/Hyperedges
        visualize_eccentricity(ecc)
        visualize_retained_ecc(ecc)
        visualize_q_percolation(Q)
        visualize_p_percolation(P)
        visualize_q_slice(structure, row_headers)
    else:
        pass
    return Qchains, Qnear, Fronts, structure, Q, P, ecc, complexity_val


def computeConjQAnalysis(matrix, row_headers, col_headers, label, visualize=None, theta=1):
    '''
    For now we have to run an inital scan of the data to discover the structure in order
    to compute a complete structural analysis of the data. This could be shortfall of the algortithm.
    '''
    verticies = col_headers
    simplicies = row_headers
    #Return variables
    Qchains = []
    Qnear = []
    Fronts = []
    #rotate the incident matrix to get the 
    incident = incidentMatrix(matrix, theta=theta)
    conj = computeConjugate(incident)
    incident = conj
    conj = computeConjugate(incident)
    #Compute the Q-analysis on the conjugate
    qface = computeQFace(incident, conj)
    qmatrix = computeQMatrix(qface)
    structure = computeQStruct(qmatrix)
    cnt = 1
    while cnt <= max(structure):
        qface = computeQMatrix(qface)
        #graph methods
        wedges = construct_weighted_graph(qface, verticies)
        #simplicial complex  methods
        qchains = computeEqClasses(wedges)
        qnear = computeQNear(wedges)
        qfronts = computeFronts(qnear)
        #Retain these attributes
        if len(wedges) < 1:
            pass
        else:
            Qchains.append(qchains)
            Qnear.append(qnear)
            Fronts.append(qfronts)
        #visualize methods
        if len(wedges) < 1:
            pass
        else:
            if visualize == True:
                dim = cnt - 1
                print('Q Dimension: ', dim)
                visualize_weighted_graph(wedges)
                print('Q Dimension: ', dim)
                visualize_conjugate(qmatrix)
                #visualize_simple_graph(incident, conj))
                #visualize_bipart_graph(bigraph)
        cnt = cnt + 1
    #Compute Q diagnostics
    Q = computeQ(Qchains)
    ##Compute P diagnostic
    P = computeP(Qchains)
    #Compute eccentricity for each simplicie/hyper-edge
    ecc =  Ecc(Qchains, verticies, structure)
    #Compute complexity of the system
    complexity_val = complexity(Q) 
    #Print results to command line for some simple diagnostic clues.
    if visualize == True:
        visualize_eccentricity(ecc)
        visualize_retained_ecc(ecc)
        visualize_q_percolation(Q)
        visualize_p_percolation(P)
        visualize_q_slice(structure, verticies)
    else:
        pass
    return Qchains, Qnear, Fronts, structure, Q, P, ecc, complexity_val


#These two functions provide methods for computing the q-analysis over multiple slicing 
#parameters and dimensions. These are used when you need to compute over all possible
#system configurations given in the hyper-edge set.
#The computeNormQ is the traiditonal Q
def computeNormalizedQ(matrix, row_headers, col_headers, label, visualize, theta=0.1):
    normalized_matrix = normPositive(matrix)
    dim = {}
    while theta < 1:
        data = computeQAnalysis(normalized_matrix, row_headers, col_headers, label, visualize=visualize, theta=theta)
        print(data)
        dex = int(theta)*10
        dim[dex] = data[0]
        theta = theta + theta
    return dim


#This function computes the conjugate complex over all slicing and dimensional levels.
def computeNormalizedConj(matrix, row_headers, col_headers, label, visualize, theta=0.1):
    normalized_matrix = normPositive(matrix)
    dim = {}
    while theta < 1:
        data = computeConjQAnalysis(normalized_matrix, row_headers, col_headers, label, visualize=visualize, theta=theta)
        dex = int(theta)*10
        dim[dex] = data[0]
        theta = theta + theta
    return dim
    

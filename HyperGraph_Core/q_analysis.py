#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:04:26 2018

@author: justinsmith
"""
import numpy as np
import networkx as nx
from hyper_graph import *
from graph_utils import *
from diagnostic import *
from visualization import *

"""
The following function represents the main app function for computing system properties
"""    
def computeQAnalysis(matrix, row_headers, col_headers, label):
    '''
    For now we have to run an inital scan of the data to discover the structure in order
    to compute a complete structural analysis of the data. This could be shortfall of the algortithm.
    '''
    simplicies = row_headers
    verticies = col_headers
    
    Qchains = []
    Qnear = []
    Fronts = []
    
    incident = incidentMatrix(matrix, theta=1)
    conj = computeConjugate(incident)
    qface = computeQFace(incident, conj)
    qmatrix = computeQMatrix(qface)
    structure = computeQStruct(qmatrix)
    
    cnt = 1
    while cnt <= max(structure):
        #matrix constructor methods
        incident = incidentMatrix(matrix, theta=cnt)
        conj = computeConjugate(incident)
        qface = computeQFace(incident, conj)
        qmatrix = computeQMatrix(qface)
        #graph methods
        wedges = construct_weighted_graph(qmatrix, simplicies)
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
            dim = cnt - 1
            print('Q Dimension: ', dim)
            print(label)
            visualize_weighted_graph(wedges)
            print('Q Dimension: ', dim)
            print(label)
            visualize_qmatrix(qmatrix)
            #visualize_bipart_graph(bigraph)
        #visualize methods
        #visualize_simple_graph(incident, conj))
        #visualize_bipart_graph(bigraph)
          
        cnt = cnt + 1
    return Qchains, Qnear, Fronts, structure


def computeConjQAnalysis(matrix, row_headers, col_headers, label):
    '''
    For now we have to run an inital scan of the data to discover the structure in order
    to compute a complete structural analysis of the data. This could be shortfall of the algortithm.
    '''
    verticies = col_headers
    simplicies = row_headers
    
    Qchains = []
    Qnear = []
    Fronts = []
    
    incident = incidentMatrix(matrix, theta=1)
    conj = computeConjugate(incident)
    
    #Swap values
    incident = conj
    conj = computeConjugate(incident)
    
    qface = computeQFace(incident, conj)
    qmatrix = computeQMatrix(qface)
    structure = computeQStruct(qmatrix)
    
    cnt = 1
    while cnt <= max(structure):
        #matrix constructor methods
        incident = incidentMatrix(matrix, theta=cnt)
        conj = computeConjugate(incident)
    
        #Swap values
        incident = conj
        conj = computeConjugate(incident)
        
        qface = computeQFace(incident, conj)
        qmatrix = computeQMatrix(qface)
        #graph methods
        wedges = construct_weighted_graph(qmatrix, verticies)
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
            dim = cnt - 1
            print('Q Dimension: ', dim)
            print(label)
            visualize_weighted_graph(wedges)
            print('Q Dimension: ', dim)
            print(label)
            visualize_conjugate(qmatrix)
            #visualize_simple_graph(incident, conj))
            #visualize_bipart_graph(bigraph)
          
        cnt = cnt + 1
    return Qchains, Qnear, Fronts, structure



def computeQAnalysis2(matrix, row_headers, col_headers, label):
    '''
    For now we have to run an inital scan of the data to discover the structure in order
    to compute a complete structural analysis of the data. This could be shortfall of the algortithm.
    '''
    simplicies = row_headers
    verticies = col_headers
    
    Qchains = []
    Qnear = []
    Fronts = []
    
    normalized_matrix = normPositive(matrix)
    theta = 0.1
    while theta < 1:
        incident = incidentMatrix(normalized_matrix, theta=theta)
        conj = computeConjugate(incident)
        qface = computeQFace(incident, conj)
        qmatrix = computeQMatrix(qface)
        structure = computeQStruct(qmatrix)
        print("Threshold Value: ", theta)
        cnt = 1
        while cnt <= max(structure):
            #matrix constructor methods
            qface = computeQMatrix(qface)
            #graph methods
            wedges = construct_weighted_graph(qface, simplicies)
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
                dim = cnt - 1
                print('Q Dimension: ', dim)
                print(label)
                visualize_weighted_graph(wedges)
                print('Q Dimension: ', dim)
                print(label)
                visualize_qmatrix(qmatrix)
                #visualize_bipart_graph(bigraph)
            #visualize methods
            #visualize_simple_graph(incident, conj))
            #visualize_bipart_graph(bigraph)
              
            cnt = cnt + 1
        theta = theta + 0.1
    return Qchains, Qnear, Fronts, structure


def computeConjQAnalysis2(matrix, row_headers, col_headers, label):
    '''
    For now we have to run an inital scan of the data to discover the structure in order
    to compute a complete structural analysis of the data. This could be shortfall of the algortithm.
    '''
    verticies = col_headers
    simplicies = row_headers
    
    Qchains = []
    Qnear = []
    Fronts = []
    
    normalized_matrix = normPositive(matrix)
    theta = 0.1
    while theta <= 1.0:
        incident = incidentMatrix(matrix, theta=theta)
        conj = computeConjugate(incident)
        
        #Swap values
        incident = conj
        conj = computeConjugate(incident)
        
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
                dim = cnt - 1
                print('Q Dimension: ', dim)
                print(label)
                visualize_weighted_graph(wedges)
                print('Q Dimension: ', dim)
                print(label)
                visualize_conjugate(qmatrix)
                #visualize_simple_graph(incident, conj))
                #visualize_bipart_graph(bigraph)
              
            cnt = cnt + 1
        theta = theta + 1
    return Qchains, Qnear, Fronts, structure

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

class hypergraph:
    
    def __init__(self,  hyperedge_set, vertex_set):
        self.hyperedge_set = hyperedge_set
        self.vertex_set = vertex_set
        
        self.normed_matrix = []
        self.incident = []
        self.incident_transpose = []

            
    #Optional: Step 0
    def normPositive(self, matrix):
        #Normalize a matrix given a raw set of values. This produces a positive value.
        #Optional transformation of input matrix given a set of raw values. This function is 
        #typically used in conjunction with a vector of weights.
        self.normed_matrix = []
        for i in matrix:
            x = max(i)
            y = min(i)
            row = []
            for j in i:
                a = (j-y)/(x-y)
                row.append(a)
            self.normed_matrix.append(row)      
        return self.normed_matrix
    
    #Optional: Step 0
    def normNeg(self, matrix):
        #Normalize a matrix given a raw set of values. This produces a negative value.
        #Optional transformation of input matrix given a set of raw values. This function is 
        #typically used in conjunction with a vector of weights. This is used when negative value
        #is desirable as a critera.
        normed_matrix = []
        for i in matrix:
            x = max(i)
            y = min(i)
            row = []
            for j in i: 
                a = (x-j)/(x-y)
                row.append(a)
            normed_matrix.append(row)      
        return normed_matrix
    
    
    # Step 1: Construct Incidence Matrix from a given matrix and threshold or slicing parameter.
    def incidentMatrix(self, matrix, theta, norm=True):
        # This function provides a basic method for describing a relation between two sets that
        # have been computed as a MxN matrix of values that map to simplicies (rows) and vertices (columns).
        # The theta value represents a threshold parameter for defining the partition of the matrix into
        # 0's and 1's.
        if norm is True:
            new_matrix = self.normPositive(matrix)
        else:
            new_matrix = matrix
            
        incident = np.zeros((len(new_matrix), len(new_matrix[0]))).astype(int)
        count = 0
        # Iterate through the rows of the matrix
        for i in matrix:
            cnt = 0
            # Iterate through the columns of the matrix
            for j in i:
                if j >= theta:
                    incident[count][cnt] = 1
                else:
                    incident[count][cnt] = 0
                cnt = cnt+1
            count = count+1
        self.incident = np.array(incident)
        return self.incident
    
    
    # Step 2: Compute the conjugate or tranpose of the retained inicidence matrix
    def computeConjugate(self):
        #Compute the conjugate of the incidence matrix
        self.incident_transpose = self.incident.transpose()
        return self.incident_transpose
    
    
    # Step 3: Multiply the incidence by its transpose - BxB^T
    # This produces the shared-faced-matrix of the incidence and its transpose
    def computeQFace(self):
        #Multiply the incidence matrix by its transpose
        shared_face_matrix = self.incident.dot(self.incident_transpose)
        return shared_face_matrix
    
    
    # Step 4: Compute the Q Matrix by subtracting 1 from each value in the matrix
    def computeQMatrix(self, shared_face_matrix):
        # compute the Q-Matrix as the shared face matrix minus E
        E = np.ones(shared_face_matrix.shape).astype(int)
        qmatrix = np.subtract(shared_face_matrix, E)
        return qmatrix
    
    
    # Step 5: Extract the Q Structure Vector from the Q Matrix
    def computeQStruct(self, qmatrix):
        #Extract the first Q Structure vector
        matrix = np.array(qmatrix)
        q_vector = matrix.diagonal()
        return q_vector

    # Step 6: This is step is simply one approach to 
    def construct_weighted_graph(self, qmatrix):
        """
        Takes the shared-face matrix q-1 computed during a q-analysis sequence.
        the ij value of a in A corresponds to the the weighted relation between two simplicies.
        """
        edges = []
        count = 0
        for i in qmatrix:
            cnt = 0
            for j in i:
                if j > 0:
                    edges.append((self.hyperedge_set[count], self.hyperedge_set[cnt], j))
                else:
                    pass
                cnt = cnt + 1
            count = count + 1
        return edges  


    # The next set of functions conpute additional attributes relevant to Q-Analysis
    def computeEqClasses(self, edges):
        #Collect all connected compoenents 
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        comp = nx.connected_components(G)
        return list(comp)
    
    
    def computeQNear(self, edges):
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        qnear = dict(nx.all_pairs_shortest_path(G))
        return qnear
    
    
    def computeFronts(self, qnear):
        t_fronts = {}
        for i in qnear.items():
            front_list = []
            fronts = defaultdict(list)
            for j in i[1].items():
                dim = len(j[1]) - 1
                fronts[dim].append(j[0])
            [front_list.append(k[1]) for k in fronts.items()]
            t_fronts[i[0]] = front_list
        return t_fronts
    
     
    def computeP(self, qstruct_set):
        pstruct = {}
        dim = 0
        for i in qstruct_set:
            val = 0
            for j in i:
                val = len(j) + val
            pstruct[dim] = val
            dim = dim + 1
        return pstruct
        
    
    def computeQ(self, qstruct_set):
        qstruct = {}
        dim = 0
        for i in qstruct_set:
            qstruct[dim] = len(i)
            dim =  dim+1
        return qstruct


    def computeEcc(self, EqClasses, qstruct):
        # eccI = 2(sum(q_dim/num_simps))/(q_dim*(q_dim+1))
        eccentricity = {}
        for simplex in self.hyperedge_set:
            simplex_dim = []
            dim = 0
            for i in EqClasses:
                for j in i:
                    if simplex in j:
                        val = dim/len(j)
                        simplex_dim.append(val)
                    else:
                        pass
                    dim = dim + 1
            ecc = sum(simplex_dim)/((1/2*float(max(qstruct)))*float((max(qstruct)+1)))
            eccentricity[simplex] = ecc  
        return eccentricity
    
    
    def computeComplexity(self, q_percolation):
        strct = []
        vect = []
        for i in q_percolation.items():
            x = i[0]+1
            y = x * i[1]
            strct.append(y)
            vect.append(i[1])
        z = sum(strct)
        complexity = 2*(z/((max(vect)+1)*(max(vect)+2)))
        return complexity


    def computeQanalysis(self, matrix, theta, norm_matrix=False, conjugate=False):
        Qchains = []
        Qnear = []
        Qfronts = []
        
        if norm_matrix is False:
            self.incidentMatrix(matrix, theta=theta, norm=False)
        else:
            self.incidentMatrix(matrix, theta=theta, norm=True)
            
        if conjugate is False:
            pass
        else:
            self.incident = self.computeConjugate()
            
        self.computeConjugate()
        shared_face_matrix = self.computeQFace()
        qmatrix = self.computeQMatrix(shared_face_matrix)
        strct = self.computeQStruct(qmatrix)
        
        count = 1
        while count < max(strct):
            shared_face_matrix = self.computeQMatrix(shared_face_matrix)
            #graph methods
            wedges = self.construct_weighted_graph(shared_face_matrix)
            #simplicial complex  methods
            qchains = self.computeEqClasses(wedges)
            qnear = self.computeQNear(wedges)
            qfronts = self.computeFronts(qnear)
            
            if len(wedges) < 1:
                pass
            else:
                Qchains.append(qchains)
                Qnear.append(qnear)
                Qfronts.append(qfronts)
                
            count = count + 1
        
        self.q_percolation = self.computeQ(Qchains)
        ##Compute P diagnostic
        self.p_percolation = self.computeP(Qchains)
        #Compute eccentricity for each simplicie/hyper-edge
        self.eccentricity =  self.computeEcc(Qchains, strct)
        #Compute complexity of the system
        self.complexity = self.computeComplexity(self.q_percolation) 
        return Qchains, Qnear, Qfronts, self.q_percolation, self.p_percolation, self.eccentricity, self.complexity 
    

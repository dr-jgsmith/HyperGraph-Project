#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 10:02:09 2018

@author: justinsmith

Global Structure Vectors
"""
def computeP(qchains):
    P = {}
    dim = 0
    for i in qchains:
        val = 0
        for j in i:
            val = len(j) + val
        P[dim] = val
        dim = dim + 1
    return P
    
def computeQ(qchains):
    Q = {}
    dim = 0
    for i in qchains:
        Q[dim] = len(i)
        dim =  dim+1
    return Q

def Ecc(EqClasses, simplex_set, strct):# eccI = 2(sum(q_dim/num_simps))/(q_dim*(q_dim+1))
    simplex_ecc = {}
    for simplex in simplex_set:
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
        ecc = sum(simplex_dim)/((1/2*float(max(strct)))*float((max(strct)+1)))
        #ecc = (2*sum(simplex_dim))/(float(max(strct)))*float((max(strct)+1))
        simplex_ecc[simplex] = ecc  
    return simplex_ecc


def complexity(qvector):
    strct = []
    vect = []
    for i in qvector.items():
        x = i[0]+1
        y = x * i[1]
        strct.append(y)
        vect.append(i[1])
    z = sum(strct)
    compl = 2*(z/((max(vect)+1)*(max(vect)+2)))
    return compl


def get_thetas(matrix):
    nums = []
    for i in matrix:
        mn = min(i)
        nums.append(mn)
        mx = max(i)
        nums.append(mx)
    thetas = list(range(min(nums), max(nums)+1))
    return thetas

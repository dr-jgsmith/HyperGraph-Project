#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:14:35 2018

@author: justinsmith

HyperGraph_Partition.py

"""


from txtprocessor import TxtProcessor
from itertools import tee
from qtwo import *
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib
import seaborn as sns

process = TxtProcessor()

def partition_txt_infile(file_name):
    f = open("data/" + file_name, 'r')
    x = [line for line in f]
    y = ' '.join(x)
    data_set = []
    q_partition = y.lower().split("\n \n \n")
    for q in q_partition:
        groups = q.split("\n \n")
        for r in groups:
            resp = r.split("\n")
            for a in resp:
                if a == "  ":
                    resp.remove(a)
                else:
                    pass
            place_question = resp[0].split(" – ")
            if len(resp) > 2:
                place = place_question[0].strip(' ')
                question = place_question[1]
                responses = resp[1:]
                data_set.append((place, question, responses))
            else:
                pass  
        #sub_parts.append(tmp)
    return data_set


def process_partion(data_set):
    new_set = []
    lemmas = []
    phrases = []
    #stop_list = ['|', '?', ':', ';', "'", 'ñ', '*', 'í', 'ï', '-', '‐', '‐‐', '–', '–new', '—', '’', 'î', 'ì', '(', ')', ',', "’re", "’s", '“', '”', '\u2028', '\uf0b7']
    for i in data_set:
        terms = []
        for j in i[2]:
            process.atomize(j)
            data = process.tokens()
            process.load_sequence(data)
            k = process.get_all_layers()
            words = data.split(' ')
            [lemmas.append(w) for w in words if len(w) > 1]
            #START this code block is only nedcessary for a particlarly ill formed text.
            #These sets of commands will need to be moved to improve reusabuliuty of the code.
            if '' in k[0]:
                k[0].remove('')
            else:
                pass
            new = []
            for l in k[0]:
                test = list(l)
                if len(test) > 1:
                    if test[0] == ' ':
                        joined = ''.join(test[1:])
                    else:
                        joined = ''.join(test)
                else:
                    pass
                phrases.append(joined)
                new.append(joined)
            #END
            terms.append(new)   
        new_set.append((i[0], i[1], i[2], terms))
    bag_of_words = sorted(set(lemmas))
    bag_of_phrases =  sorted(set(phrases))
    return new_set, bag_of_words, bag_of_phrases
    


"""
The next steps in this process is to construct a simplex of simplicies

The approach taken here computes a collection of simplicies and 
"""
def computeAll(data_set, bag_of_phrases):
    phrase_set = bag_of_phrases_ids(bag_of_phrases)
    simplex = []
    for i in data_set:
        row = computeSimplex(i[3], phrase_set)
        simplex.append(row)
    return simplex
    

def computeSimplex(phrases, phrase_set):
    row = []
    simplex = flatten_simplex(phrases)
    for i in phrase_set.items():
        score = 0
        test_set = i[0].split(' ')
        for j in simplex:
            eval_set = j.split(' ')
            score =  score + simplex_similarity(test_set, eval_set)
        row.append(score)
    return row
    

def simplex_similarity(simplex_set1, simplex_set2, theta=0.20):
    coef = 0
    test_set = set(simplex_set1)
    eval_set = set(simplex_set2)
    simplex_sim = len(test_set.intersection(eval_set))/len(test_set.union(eval_set))
    if simplex_sim > theta:
        coef = simplex_sim
    else:
        pass 
    return coef
    
                 
def flatten_simplex(phrases):
    flatten = []
    for i in phrases:
        for j in i:
            flatten.append(j)
    return flatten
    
    
def bag_of_phrases_ids(bag_of_phrases):
    phrase_set = {}
    for p in bag_of_phrases:
        phrase_set[p] = bag_of_phrases.index(p)
    return phrase_set


def bag_of_words_ids(bag_of_words):
    word_set = {}
    for p in bag_of_words:
        word_set[p] = bag_of_words.index(p)
    return word_set


def computeSetsId(data_set):
    sets = []
    for i in data_set:
        sets.append((i[0], i[1]))
    return sets


def computeThemes(data_set, word_list, s):
    data_table = []
    sets = computeSetsId(data_set)
    row_id = 0
    for i in s:
        data = np.array(i)
        dev = data.mean() + 2.0 * data.std()
        topics = []
        for j in i:
            if j > dev:
                dex = i.index(j)
                topics.append(word_list[dex])
            else:
                pass
        data_table.append((sets[row_id][0], sets[row_id][1], data.mean(), data.std(), dev, sorted(set(topics))))
        row_id = row_id + 1
    return data_table
    

def computeTopics(data_table):
    words = []
    data = []
    for i in data_table:
        row = []
        for j in i[5]:
            x = j.split(' ')
            for k in x:
                words.append(k)
                row.append(k)
        
        wdata = {}
        rset = set(row)
        for l in rset:
            cnt = row.count(l)
            wdata[l] = cnt
        data.append((i, row, wdata))
           
    matrix = []
    for i in data:
        test_set = set(i[1])
        row = []
        for j in data:
            eval_set = set(j[1])
            score = len(test_set.intersection(eval_set))/len(test_set.union(eval_set))
            row.append(score)
        matrix.append(row)
    return matrix
    

def findXTopic(matrix, data_set):   
    data = {}
    row_id = 0
    for i in matrix:
        count = 0
        tid = []
        for j in i:
            if j > 0.2:
                tid.append((data_set[count][0], data_set[count][1], j))
            else:
                pass
                #tid.append(j)
            count = count + 1
        
        data[row_id] = tid
        row_id =  row_id + 1
    return data
                
    

def computeTopicMatrix(matrix):
    data = []
    word_uni = []
    for i in matrix:
        obs_words = []
        for j in i[5]:
            words = j.split(' ')
            for w in words:
                if w == 'a':
                    pass
                else:
                    word_uni.append(w)
                    obs_words.append(w)
        data.append((i[0], i[1], i[5], obs_words))
    
    word_set = sorted(set(word_uni))
    final = []
    for i in data:
        row = []
        N = [0] * len(word_set)
        for j in i[3]:
            loc = word_set.index(j)
            N[loc] = N[loc] + 1
        row.append((i, N))
        final.append(row)
    return final, word_set
        


def computeMatrix(final, word_set):
    matrix = []
    topics = []
    for i in final:
        for j in i:
            row = (j[0][0], j[0][1])
            topics.append(row)
            matrix.append(j[1])
    return matrix, topics
        
    
def computePartial(matrix, simplex_labels, label="three action steps toward ccl "):
    subset_data = []
    subset_labels = []
    count = 0
    for i in simplex_labels:
        if i[1] == label:
            subset_data.append(matrix[count])
            subset_labels.append(i[0])
        else:
            pass
        count = count + 1
    return label, subset_data, subset_labels
        

def get_labels(labels):
    labels_id = {}
    for i in labels:
        labels_id[labels.index(i)] = i
    return labels_id

'''
x = partition_txt_infile('summit_text.txt')
print("Processing file...")

y = process_partion(x)
s = computeAll(y[0], y[2])

setids = computeSetsId(y[0])
themes = computeThemes(y[0], y[2], s)
print("Computed themes: ")

final = computeTopicMatrix(themes)
matrix =  computeMatrix(final[0], final[1])

questions = [i[1] for i in setids]
qset = sorted(set(questions))
    
for i in qset:
    data = computePartial(matrix[0], matrix[1], label=i)
    labels = get_labels(data[2])
    #qchains = computeConjQAnalysis(s, y[2]
    qchains = computeQAnalysis(data[1], data[2], final[1])
    
    Q = computeQ(qchains[0])
    P = computeP(qchains[0])
    
    ecc =  Ecc(qchains[0], data[2], qchains[3])
    visualize_eccentricity(ecc)
    print(ecc)
    
    visualize_q_percolation(Q)
    print(Q)
    visualize_p_percolation(P)
    print(P)
    visualize_q(qchains[3], data[2])
    visualize_q_slice(qchains[3], data[2])
    
    topic_themes = []
    for k in themes:
        if k[1] == i:
            [topic_themes.append(j) for j in k[5]]
        else:
            pass
    
    phrase_counts = {}
    test = sorted(set(topic_themes))
    for m in test:
        val = topic_themes.count(m)
        phrase_counts[m] = val
    new_list = {}
    for n in phrase_counts.items():
        if n[1] > 2:
            new_list[n[0]] = n[1]
        else:
            pass
    print(new_list)
    #Mainly for computing over the conjugate
    #visualize_retained_ecc(ecc)
'''
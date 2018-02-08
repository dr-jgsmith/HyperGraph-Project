#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 22:05:39 2017

@author: justinsmith
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 19:28:52 2017

@author: justinsmith

This program implements a model for computing the hypernetwork for ad-hoc ontology generation.

This approach applies the algebraic topology of hypernetwork and q-analysis to compute the structure of connected
knowledge objects in a given knowledge base. For this experiment we take an input vector F that represents the features. 
In this example, we will focus primarily on the sets of noun sequences, and subject, object, predicate triples contained in a text. 
However, any number of objects could be considered, and there are numerous different ways to encode text.

For example, using the spacy.io nlp package we can isolate the sequence of nouns in a text. Given W as the set of all known nouns in the universe,
perform intersection on W and di of D, defined as the set of documents or communicative acts. Here, W is the set of all features in the universe.
and D is the set of all speech acts. In this case we want to cluster, speech acts which not only include spoken discourse but any represenation of communication
through text, audio, visual. In this example, the data stream will be text from a collection of articles.

"""
import numpy as np
import csv


from HtmlMapper import HtmlMapper

from itertools import tee
import networkx as nx
import matplotlib.pyplot as plt
import json

learn = HtmlMapper()




def encode_context(topic):
    fset.get_universe()
    fset.get_topics()
    fset.get_structure()
    """
    Note: this function provides a general utility for encoding a string sequence as an index and heirachical graph 
    
    The first step in this process is to construct a context vector or assemblage for any given word.
    We will use wikipedia link lists to encode an initial set of features
    for topic classification. Output structure is a list that contains a list with the topic vector, and another list of 
    
    ***
    Important to Note: features are also topic classes in the case of wikipedia.
    It will be important to test some additional experiments in order to test cluster
    accuracy.
    ***
    
    This function collects the link phrase from a wiki page. 
    Adds the text phrase to the master dictionary.
    Returns the index value for the phrase.
    
    Splits the phrase into single words and returns the index value for each word as an ordered list.
    
    The original index value is added to a tuple as the first object in the sequence and followed by the 
    list of partioned sequence representing the index value for each term in the phrase.
    
    This is important as 
    
    """
    context_vector = {}
    #first we will read the main topic class and add it to the dictionary.
    #this returns an index for the value of the topic class
    poly = fset.update_universe(topic.lower())[1]
    #next we we want to add the topic class to our context set.
    #in this case we split the original text
    chain = [fset.update_universe(i)[1] for i in topic.lower().split(' ')]
    fset.update_topics(poly, tuple(chain))
    
    learn.get_page(topic)
    
    #tmp = []
    try:
        data = encode_concept_from_wiki(learn.page.content, learn.page.links)
        """
        
        The structure of the returned vector sequence: var data
         {2202: (2200, 678, 727, 5, 728, 2201, 1052, 1053)}
        
        The purpose for this data structure is the ability to do subsequence set matching.
        
        """
        fset.vector_verse[poly] = tuple(data)
        fset.save_universe(fset.universe)
        fset.save_topics(fset.topics)
        fset.save_structure(fset.vector_verse)
        #fset.update_structure(poly, context_vector)
        
        print(fset.vector_verse.get(poly))
        return tuple(learn.page.links)
    except TypeError:
        pass
     

def encode_concept_from_wiki(content, wikilinks):
    chunk = ChunkText()
    process.atomize(content)
    process.tokens()
    i = process.pos_tags_c()
    print(i)
    
    d = chunk.parse(i)
    chunk.np_sub_chunks(d)
    retained = chunk.return_sub_chunks()
    [retained.append(j) for j in wikilinks]
    print(len(retained))
    print(retained)
    encoded = []
    for j in retained:
        l = j.lower().split(' ')
        a = [fset.update_universe(k)[1] for k in l]#fset.update_universe(j.lower())
        v = fset.update_universe(j.lower())[1]
        fset.update_topics(v, tuple(a))
        encoded.append(v)
    return sorted(set(encoded))
    


def new_encoder(content):
    partition = content.lower().split("\n\n\n== ")
    new_p = [i.split("\n") for i in partition]
    data = []
    for k in new_p:
        for l in k:
            tmp = []
            process.atomize(l)
            process.tokens()
            i = process.pos_tags_c()
            tmp = [(1, j[0]) if j[1] in ["ADJ", "NOUN"] else (0, j[0]) for j in i]
            seq = iteritem(tmp)
            rips = k_ripps(seq)
            g = build_graph(rips)
            flat = flatten(g) #visualize(g)
            words = [' '.join(m) for m in rips]
            data.append(words)
            data.append(flat)
    return data


def iteritem(sequence):
    pool = []
    tmp = []
    for i in sequence:
        if i[0] > 0:
            tmp.append(str(i[1]))
        else:
            pass
        pool.append(tuple(tmp))
        tmp = []
    return pool
            

def k_ripps(sequence):
    tmp = []
    pool = []
    for i in sequence:
        try:
            if i[0]:
                tmp.append(str(i[0]))
        except IndexError:
            if len(tmp) == 0:
                pass
            else:
                pool.append(tmp)
                tmp = [] 
    return pool


def build_graph(sequence):
    edges = []
    
    for i in sequence:
        edge = pairwise(i)
        #print(hold, list(edge))
        edges.append(list(edge))
    
    dex = []
    wsubg = []
    count = 0
    for j in edges:
        subg = []
        tmp = []
        dex.append(count)
        
        tmp = [l for k in j for l in k]
        subg = [(count, i) for i in set(tmp)]
        [subg.append(k) for k in j]
        #print(subg)
        count = count +1
        if len(subg) > 1:
            wsubg.append(subg)
        else:
            pass
        
    v_dex = pairwise(dex)  
    [wsubg.append(i) for i in list(v_dex)]
    return wsubg
                       


def flatten(wsubg):
    data = []
    for i in wsubg:
        if len(i) > 2:
            for j in i:
                data.append(j)
        else:
            data.append(i)
    return data
   
    
    
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)    
            
            



def encode_structure(seed_term):
    """
    The next step in constructing our context vector set is to learn the relations between linked terms.
    For this we will build out a recursive function that learns the context of contexts.
    
    You can define how deep this goes by altering the 'depth' parameter.
    """
    seen = [seed_term]
    links = encode_context(seed_term)
    for i in links:
        print(i)
        if i in seen:
            pass
        else:
            encode_context(i)
            seen.append(i)
    
    return seen



def sent_phrases(input_text):
    """
    The sent_phrases method partitions text by sentence and by word into a 
    heirachical bag-of-words representation. 
    
    In this context, a sentence in a text serves as a kind of feature in a given text from which
    pattern matching algorithms are applied to determine the presence of a sub-set from the total feature set.
    
    
    The partions of text utterances into sentences is, I believe an important structural
    necessity in developing a useful language model. 
    
    While I was orginally thinking I would focuson a bag-of-words representation,  the layered or multi-level aspect of having, words,
    phrases, sentences, paragraphs, and documents structured heirachically provides a means
    to capture a richer dewxcription of the polyhedra in and between text.
    
    This approach organizes text entities cover sets takes an explicit different approach in the construction 
    
    """
    fset.get_universe()
    data = []
    process.atomize(input_text)
    process.sent_tokens() #this method partions the text based upon a '. ' space pattern. Other sentence partion methods might be useful.
    result = process.pos_tags_s()
    for i in result:
        tmp = []
        tmp = [fset.update_universe(str(j[0]))[1] for j in i]
        data.append(tuple(tmp))
        
    fset.save_universe(fset.universe)
    return data


def sent_phrases_chunks(input_text):
    chunk = ChunkText()
    """
    The sent_phrases method partitions text by sentence and by word phrases that make up each sentence.
    
    In this context, a sentence in a text serves as a kind of feature in a given text from which
    pattern matching algorithms are applied to determine the presence of a sub-set from the total feature set.
    
    
    The partions of text utterances into sentences is, I believe an important structural
    necessity in developing a useful language model. 
    
    While I was orginally thinking I would focuson a bag-of-words representation,  the layered or multi-level aspect of having, words,
    phrases, sentences, paragraphs, and documents structured heirachically provides a means
    to capture a richer dewxcription of the polyhedra in and between text.
    
    This approach organizes text entities cover sets takes an explicit different approach in the construction 
    
    """
    data = []
    
    process.atomize(input_text)
    result = process.sent_tokens() #this method partions the text based upon a '. ' space pattern. Other sentence partion methods might be useful.
     
    for i in result:
        tmp = []
        d = chunk.parse(i)
        chunk.np_sub_chunks(d)
        retained = chunk.return_sub_chunks()
        for j in retained:
            d = fset.update_universe(str(j))[1]
            fset.update_topics(d, tuple(j.split(' ')))
            tmp.append(d)
        
        data.append(tuple(tmp))
    
    '''
    The structure of the returned vector sequence: var data
    
    
    The purpose for this data structure is the ability to do subsequence set matching.
    '''
    return data



def simple_matcher(sent_tokens, threshold=0.75):
    simple_set = []
    fset.id2sequence()
    fset.get_structure()
    #Call topics in tuple form, i.e. the partioned sequences in ID form
    b = fset.topic_seqs.keys()
    #Generate the polyhedra as topic items in list form / might be an unnecessary step.
    polyhedra = [list(i) for i in b]
    for i in polyhedra:
        #for each subsequence in polyhedra see if there is a match with the token set
        for j in sent_tokens:
            #calculate similarity
            intersect = set(i) & set(j)
            #union = set(i) | set(j)
            index = len(intersect)/len(i)
            if index > threshold:
                e = fset.topic_seqs.get(tuple(i))
                simple_set.append((e, index))
            else:
                pass
    '''
    This returns an array to be constructed into a pandas dataframe for further analysis.
    '''
    data = [(i[0], simple_set.count(i)) for i in simple_set]
    return sorted(set(data))
    


def less_simple_matcher(sent_tokens, threshold=0.82):
    simple_set = []
    fset.id2sequence() #This returns the current topic list in their atomic 
    fset.get_structure()
    #Call topics in tuple form, i.e. the partioned sequences in ID form
    b = fset.topic_seqs.keys()
    print(b)
    #Generate the polyhedra as topic items in list form / might be an unnecessary step.
    polyhedra = [list(i) for i in b]
    for i in polyhedra:
        #for each subsequence in polyhedra see if there is a match with the token set
        for j in sent_tokens:
            #calculate similarity
            intersect = set(i) & set(j)
            #union = set(i) | set(j)
            #get a similarity score between sets
            index = len(intersect)/len(i) #+len(j))
            if index > threshold:
                e = fset.topic_seqs.get(tuple(i))
                test = fset.vector_verse.get(tuple(i))
                if str(e) in str(test):
                    index = index * 100
                else:
                    index = index
                simple_set.append((e, index))
            else:
                pass
    '''
    This returns an array to be constructed into a pandas dataframe for further analysis.
    '''
    data = [(i[0], simple_set.count(i)) for i in simple_set]
    return sorted(set(data))






def match_context(matched_topics):
    reverse = {}
    topics = []
    data = []
    fset.get_structure()
    for i in fset.vector_verse.items():
        reverse[i[1]] = i[0]
    
    contexts = list(reverse.keys())
    xs = [i[0] for i in matched_topics]
    
    for i in contexts:
        tmp = []
        intersect = set(i) & set(xs)
        if len(intersect) > 0:
            for j in matched_topics:
                if j[0] in intersect:
                    tmp.append(j)
                else:
                    pass
            #t = [x for j in matched_topics for x in j if x[0] in sorted(intersect)]
            #topics.append((t, len(xs), len(i), len(intersect)))  
            topics.append((reverse.get(i), len(intersect), len(xs), len(i)))
        else:
            pass
        data.append(tmp)
    return topics
        

def rank_topics(topic_tuples):
    ranks = []
    for i in topic_tuples:
        sim = i[1]/(i[2]+i[3])
        ranks.append((i[1], sim, fset.lookup_word(i[0]), i[0], i[2]))
    return sorted(ranks)



def context_weights():
    reverse = {}
    N = fset.get_universe()
    for i in N.items():
        reverse[i[1]] = i[0]
    
    fset.get_structure()
    data = fset.vector_verse.keys()
    
    tmp =[]
    rows = []
    for i in data:
        rows.append(i)
        n = [0] * len(N)
        for j in fset.vector_verse.get(i):
            n[j-1] = n[j-1] + 1
        tmp.append(n)
    
    matrix = np.array(tmp)
    A = matrix
    B = A.transpose()
    R = A.dot(B)
    E = np.ones(R.shape)
    S = R - E
    D = S.diagonal()
    
    '''
    A = pd.DataFrame(matrix, index=rows, columns=list(fset.universe.keys()))
    B = A.T
    R = A.dot(B)
    E = pd.DataFrame(np.ones(R.shape), index=rows, columns=rows)
    S = R.subtract(E)
    S.to_csv('data/wiki_graph.csv')
    QM = np.array(S)
    q_vector = QM.diagonal()
    '''
    
    return S
    
    
    
def process_docs_from_csv(filename):
    fset.get_structure()
    file = open(filename, 'r', errors='ignore')
    outfile = open('data/corpus_graph_II.csv', 'w', newline='')
    rows = csv.reader(file)
    new = ['id', 'title', 'source_url', 'phrases', 'graph']
    outd = csv.writer(outfile)
    edge = ['source', 'target']
    outd.writerow(edge)
    count = 0
    for row in rows:
        data = new_encoder(row[5])
        #data_row = [row[0], row[2], row[4], data[0], data[1]]
        print(data[1])
        for i in data[1]:
            edge_row = [i[0], i[1]]
            print(edge_row)
            outd.writerow(edge_row)
        count = count+1
        print(count)
        
process_docs_from_csv('data/ArticlesDBRecords.csv')
#encode_structure('agriculture')
'''
links = learn.get_links('community development')
x = links
for i in x:
    encode_context(i)
          
x =  learn.get_page('climate change')
y = sent_phrases(x)
print(y)

z = simple_matcher(y)
print(z)
df = pd.DataFrame(z)
print(df)

for i in docs_list:
    j = sent_phrases(i)
    z = simple_matcher(j)
    d = pd.DataFrame(z)
    df.append(d)
'''
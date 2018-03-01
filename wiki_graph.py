#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 00:35:06 2018

@author: justinsmith
"""
from web_data import HtmlMapper
from hyperdb import hyperdb
from txtprocessor import TxtProcessor
#import time

learn = HtmlMapper()
process =  TxtProcessor()
db = hyperdb() #HyperGraphDB('data/wiki_graph.db')
db.create_db('training_data/wiki_graph2.csv', 'title', 'hyperedge', 'vertex', 'value')

def encoder(content):
    if content:
        norm = content.lower()
        process.atomize(norm)
        clean = process.tokens()
        process.load_sequence(clean)
        phrases = process.noun_phrase_re()
        return phrases
    else:
        return None
    

def start_search(topic):
    content = learn.get_page(topic)
    data = encoder(content)

    topics = learn.page.links
    related_topics = [i.lower() for i in topics]
    features = data + related_topics
    bag_of_words = []
    for i in features:
        x = i.split(' ')
        [bag_of_words.append(j) for j in x]
    
    bow = sorted(set(bag_of_words))
    bow_counts = {}
    for k in bow:
        bow_counts[k] = bag_of_words.count(k)
    
    vertices = sorted(set(features))
    
    counts = {}
    for vertex in vertices:
        
        x = vertex.split(' ')
        value = []
        for j in x:
            val = bow_counts.get(j)
            value.append(val)
        
        counts[vertex] = str(sum(value)/len(x))
        db.add_hyperedge('training_data/wiki_graph2.csv', 'wiki_graph', topic, vertex, str(sum(value)/len(x)))
        #print(l, str(sum(value)/len(x)))
    return topics
    

def deep_search(topic, depth):
    count = 0
    seen = []
    topics = [topic]
    while count < depth:
        for i in topics:
            #time.sleep(1)
            if i in seen:
                topics.remove(i)
                pass
            else:
                data = start_search(i)
                seen.append(i)
                topics.extend(data)
            col = db.get_col('hyperedge')
            print(i, len(set(col)))
        count = count + 1
                

def search_list(topic_list):
    topics = []
    seen = []
    for i in topic_list:
        if i in seen:
            pass
        else:
            data = start_search(i)
            topics.append(data)
            print(i)
            seen.append(i)
            topics.extend(data)
          
    topic_set = []
    for k in topics:
        for l in k:
            topic_set.append(l)
       
    for j in topic_set:
        if j in seen:
            pass
        elif len(j) < 2:
            pass
        else:
            start_search(j)
            print(j)
            seen.append(j)

    return
            
        
topic_list = ['finance', 'economics', 'disaster', 'famine', 'commercial fishing',
              'commodities', 'agriculture', 'sustainability', 'petroleum',
              'climate change', 'cryptocurrency', 'health', 'forestry', 'mining']

search_list(topic_list)
#deep_search('climate change', 1)
#y = db.get_representation('wiki_graph.csv', 'hyperedge', 'vertex')

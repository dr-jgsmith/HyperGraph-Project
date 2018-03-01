#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 00:35:06 2018

@author: justinsmith

The script is an example application for creating knowledge representations
from Wikipedia articles and link lists.

"""
from web_data import HtmlMapper
from hyperdb import hyperdb
from txtprocessor import TxtProcessor
#import time

learn = HtmlMapper()
process =  TxtProcessor()
db = hyperdb() 


def phrase_ranker(content):
    if content:
        norm = content.lower()
        data = norm.replace('\n', '')
        process.atomize(data)
        clean = process.tokens()
        process.load_sequence(clean)
        ranked_phrases = process.rank_phrases()
        return ranked_phrases
    else:
        return None

def word_ranker(content):
    if content:
        norm = content.lower()
        data = norm.replace('\n', ' ')
        process.atomize(data)
        clean = process.tokens()
        process.load_sequence(clean)
        ranked_words = process.rank_words()
        return ranked_words
    else:
        return None


def rank_phrases(topic):
    #Get wikipedia page content for a given topic
    content = learn.get_page(topic)
    #Encode topic using the word_encoder or the phrase_encoder
    data = phrase_ranker(content)
    if data:
        for i in data.items():
            #data_row = ['wiki_graph', topic, i[0], i[1]]
            db.add_hyperedge('wiki_graph', topic, i[0], i[1])
        return learn.page.links
    else:
        return None


def rank_words(topic):
    #Get wikipedia page content for a given topic
    content = learn.get_page(topic)
    #Encode topic using the word_encoder or the phrase_encoder
    data = word_ranker(content)
    if data:
        for i in data.items():
            #data_row = ['wiki_graph', topic, i[0], i[1]]
            db.add_hyperedge('wiki_graph', topic, i[0], i[1])
        return learn.page.links
    else:
        return None



def search_list(topic_list):
    topics = []
    for topic in topic_list:
        print(topic)
        data = rank_words(topic)
        [topics.append(link.lower()) for link in data]
        
    topic_set = sorted(set(topics))
    for j in topic_set:
        print(j)
        rank_words(j)
    
    return



topic_list = ['finance', 'disaster', 'famine', 'commercial fishing',
              'commodities', 'agriculture', 'sustainability', 'petroleum',
              'climate change', 'blockchain', 'health', 'forestry', 'mining']

search_list(topic_list)

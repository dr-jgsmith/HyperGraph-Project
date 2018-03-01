#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 12:26:05 2017

@author: justinsmith

These functions provide simple interfaces to basic language processing methods. 
Other NLP systems probably provide better/faster preprocessing methods. However, these methods
provide simple tokenization method, token count, and bag-of-words. 

The "Preprocessor" class also exposes the Spacy NLP framework. Additional methods could be extended
that make greater use of Spacy. Ideally, if not related to a preprocessing use case, it would be suggested
to create an additional class for defining the Spacy based methods.

Currently, the "Preprocessor" class offers two different POS tagging methods.

"""
import csv
import spacy


class TxtProcessor:
    
    def __init__(self):
        
        self.nlp = spacy.load('en')
        file = open('data/stopwords.csv', 'r')
        rfile = csv.reader(file)
        self.stop_words = [j[0] for j in rfile]
        
    """
    The methods below are a list of helper functions to split up a 
    stream of text. 
    
    """
    def atomize(self, text):
        #Get a stream of text
        self.text = text
        #break the data into their lowest level - single character
        self.txt_atoms = [c for c in text]
        #convert that string to data char to a unicode representation
        self.uni_atoms = [ord(c) for c in self.txt_atoms]
        return
        
    def tokens(self):
        #define characters that need removed
        stop_words = []
        stop_char = ['=']#['|', '?', ':', ';', "'"]
        clean = [i for i in self.txt_atoms if i not in stop_char]
        cleanwords = ''.join(clean)
        text_list = [i for i in cleanwords if i not in stop_words]
        self.cleanwords = ''.join(text_list)
        return self.cleanwords
    
    def load_sequence(self, text):
        self.data = self.nlp(text)
        return self.data
    
    def tokenize(self):
        self.tokens = [word.text for word in self.data]
        return self.tokens
    
    def pos_tags(self):
        self.tags = [(word.text, word.pos_) for word in self.data]
        return self.tags
    
    def word_roots(self):
        self.roots = [word.lemma_ for word in self.data]
        return self.roots

    
    def entities(self):
        self.entity = [(word.text, word.label_) for word in self.data.ents]
        return self.entity
        
    def parse_triples(self):
        self.trigram = []
        for np in self.data.noun_chunks:
            struct = (np.root.text, np.root.dep_, np.root.head.text)
            self.trigram.append(struct)
        return self.trigram
    
    def noun_phrase(self):
        phrase = []
        for np in self.data.noun_chunks:
            struct = (np.text)
            phrase.append(struct)
        return phrase
            
    def noun_phrase_re(self):
        self.clean_phrases = []
        phrase = []
        #stop_words = ['a', '=', '-', 'the', 'it', 'its', 'i', 'we', 'ñ', 'í', 'ï', "it's", 'them', 'this', 'they', 'there', 'these', 'their', 'in', 'on', 'be', 'at']
        for np in self.data.noun_chunks:
            struct = (np.text)
            phrase.append(struct)
        for i in phrase:
            data = i.split(' ')
            for j in data:
                if j in self.stop_words:
                    data.remove(j)
                else:
                    pass
            cdata = ' '.join(data)
            self.clean_phrases.append(cdata)
        return self.clean_phrases
            
    
    def get_all_layers(self):
        phrases = self.noun_phrase_re()
        roots = self.word_roots()
        tags = self.pos_tags()
        return phrases, roots, tags


     
def convert_text(file_name):
    f = open("data/" + file_name, 'r')
    x = []
    for line in f:
        x.append(line)
    y = ' '.join(x)
    return y      
        
'''
ptext = TxtProcessor()
x = convert_text('summit_text.txt')
ptext.atomize(x)
ptext.tokens()
y = ptext.load_sequence()
allseq = ptext.get_all_layers()
print(allseq[0])
'''
        
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 12:26:05 2017

@author: justinsmith

These functions provide simple interfaces to basic language processing methods offered
by the Spacy.io NLP Framework. Other NLP systems may provide better/faster preprocessing
methods. However, these methods provide simple tokenization method, token count, and 
bag-of-words. 

"""
import csv
import spacy


class TxtProcessor:
    
    def __init__(self):
        #Initialize a Spacy object
        self.nlp = spacy.load('en')
        #If not using Spacy builtin stopwords list, use this file to create
        #a stopwords list. Note: may be removed in future versions.
        file = open('data/stopwords.csv', 'r')
        rfile = csv.reader(file)
        self.stop_words = [j[0] for j in rfile]
        
    #The methods below are a list of helper functions to split up a stream of text. 
    
    #Atomize and tokens are optional standalone tools. They can be used
    #with or without the Spacy.io framework.
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
        stop_char = ['=', '(', ')', '/', ':', ';', ',', '?', '|' '"', '[', ']']#['|', '?', ':', ';', "'"]
        clean = [i for i in self.txt_atoms if i not in stop_char]
        cleanwords = ''.join(clean)
        text_list = [i.replace('.', ' ')  for i in cleanwords if i not in self.stop_words]
        self.cleanwords = ''.join(text_list)
        return self.cleanwords
    
    #Primary Spacy.io based methods for basic text processing
    #Steps include load text or sequence
    def load_sequence(self, text):
        self.data = self.nlp(text)
        return self.data
    
    #Tokenization and Stop Word Removal
    def tokenize(self):
        self.tokens = [word.text for word in self.data if word.is_stop == False]
        return self.tokens
    
    #Parts-of-Speech Tagging
    def pos_tags(self):
        self.tags = [(word.text, word.pos_) for word in self.data]
        return self.tags
    
    #Lemmatize a sequence of words - Get root word
    def word_roots(self):
        self.roots = [word.lemma_ for word in self.data if word.is_stop == False]
        return self.roots
    
    #Extract named entities from a text
    def entities(self):
        self.entity = [(word.text, word.label_) for word in self.data.ents]
        return self.entity
        
    #Extract Subject, Object, Predicate
    def parse_triples(self):
        self.trigram = []
        for np in self.data.noun_chunks:
            struct = (np.root.text, np.root.dep_, np.root.head.text)
            self.trigram.append(struct)
        return self.trigram
    
    #Simple noun-phrase extraction
    def noun_phrase(self):
        phrase = []
        for np in self.data.noun_chunks:
            struct = (np.text)
            phrase.append(struct)
        return phrase
            
    #More complex noun-phrase extraction method
    def noun_phrase_re(self):
        clean_phrases = []
        phrase = []
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
            clean_phrases.append(cdata)
        return clean_phrases
            
    #Rank words by occurrence in a text
    def rank_words(self):
        roots = list(self.word_roots())
        root_set = sorted(set(roots))
        counts = {}
        for i in root_set:
            val = roots.count(i)
            counts[i] = val
        return counts
    
    #Rank phrases based on occurrence of phrase words throughout a text.
    def rank_phrases(self):
        #Count roots words in the phrase
        lemma_counts = {}
        #Use simple phrase extractor
        phrases = self.noun_phrase()
        #Get the entire set of words and their lemmas
        roots = list(self.word_roots())
        #Count lemmas in the text
        lemma_set = sorted(set(roots))
        for lem in lemma_set:
            val = roots.count(lem)
            lemma_counts[lem] = val
        #Collect phrases    
        ranked_phrases = {}
        for j in phrases:
            #load phrase
            self.load_sequence(j)
            #root words in the phrase
            proots = self.word_roots()
            #Check to see if the phrase is an empty string
            if len(proots) > 0:
                #count phrase lemmas in the text
                value = [lemma_counts.get(k) for k in proots]
                if None in value:
                    pass
                else:
                    #connect lemmas to reconstruct phrase
                    phrase = ' '.join(proots)
                    #sum occurrences of lemmas in text and apply value to phrase
                    ranked_phrases[phrase] = str(sum(value)/len(proots))
            else:
                pass    
            
        return ranked_phrases
            
    #Simply gets the phrases, lemmas and parts-of-speech tags for a sequence of text.            
    def get_all_layers(self):
        phrases = self.noun_phrase_re()
        roots = self.word_roots()
        tags = self.pos_tags()
        return phrases, roots, tags
        
        

        
        
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:16:36 2016

@author: admin
"""
import nltk
from .utils import featid

class LexicalFeatures:
    
    def __init__(self, lof):
        self.lof = lof
    
    def computeDensity(self, taggedSentences):
        densities = []
        #jnrv = ['J', 'N', 'R', 'V'] # nouns, adjectives, adverbs or verbs. 

        for sent in taggedSentences:
            jnrv = [word[1] for word in sent
                    if word[1].startswith('J') or word[1].startswith('N')
                    or word[1].startswith('R') or word[1].startswith('V')]
            densities.append(float(len(sent) - len(jnrv)) / len(sent))
        
        return densities

    @featid(3)        
    def lexicalDensity(self, argString):
        '''
        The frequency of tokens that are not nouns, adjectives, adverbs or verbs. 
        This is computed by dividing the number of tokens tagged with POS tags 
        that do not start with J, N, R or V by the number of tokens in the chunk
        '''
        density = []
        taggedSents = []
        for sentence in self.lof:
            tokens = nltk.word_tokenize(sentence)
            taggedSents.append(nltk.pos_tag(tokens))

        return self.computeDensity(taggedSents)

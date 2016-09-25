# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:19:12 2016

@author: admin
"""
import codecs
import nltk

class Preprocess:
    
    fileName = ''
    
    def __init__(self,fileName):
        self.inputFile = fileName
        self.plainLof = []
        self.tokenSents = []
        self.nltkPOSSents = []

    def preprocessBySentence(self):

        with codecs.open(self.inputFile, encoding='utf-8') as f:
            lines = f.read().splitlines()
        return lines

    def preprocessClassID(self):
        """ Extract from each line the integer for class ID."""

        with codecs.open(self.inputFile, encoding='utf-8') as f:
            lines = f.read().splitlines()
        ids = [int(id) for id in lines]
        
        return ids

    def preprocessByBlock(self, fileName, blockSize):
        pass

    def getPlainSentences(self):
        if not self.plainLof:
            self.plainLof = self.preprocessBySentence()
        return self.plainLof

    def gettokenizeSents(self):
        if not self.tokenSents:
            self.tokenSents = [nltk.word_tokenize(sent) for sent in self.plainLof]
        return self.tokenSents

    def nltkPOStag(self):
        """ Tag given sentences with POS of nltk. """
        if not self.nltkPOSSents:
            if not self.tokenSents:
                self.gettokenizeSents()
            self.nltkPOSSents = [nltk.pos_tag(tokens) for tokens in self.tokenSents]

        return self.nltkPOSSents

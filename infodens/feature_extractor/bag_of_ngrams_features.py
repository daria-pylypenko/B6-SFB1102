# -*- coding: utf-8 -*-
"""
Created on Sun Sep 04 14:12:49 2016

@author: admin
"""
import ast

from .feature_extractor import featid, Feature_extractor
from collections import Counter
from nltk import ngrams
from scipy import sparse


class Bag_of_ngrams_features(Feature_extractor):

    def ngramArgumentCheck(self, argString, type):
        status = 1
        n = 0
        freq = 0
        filePOS = 0 # input file
        ngrams_dict = 0 # predefined list of ngrams

        argStringList = argString.split(',')
        if argStringList[0].isdigit():
            n = int(argStringList[0])
        else:
            print('Error: n should be an integer')
            status = 0
        if len(argStringList) > 1:
            if argStringList[1].isdigit():
                freq = int(argStringList[1])
            else:
                print('Error: frequency should be an integer')
                status = 0
            #POS file
            if type == "POS":
                if int(argStringList[2]):
                    filePOS = argStringList[4]
                if int(argStringList[3]):
                    ngrams_file = argStringList[4+int(argStringList[2])]
        else:
            freq = 1
        return status, n, freq, filePOS, ngrams_file

    def preprocessReqHandle(self, type, filePOS):
        if type == "plain":
            listOfSentences = self.preprocessor.gettokenizeSents()
        elif type == "POS":
            listOfSentences = self.preprocessor.getPOStagged(filePOS)
        elif type == "lemma":
            listOfSentences = self.preprocessor.getLemmatizedSents()
        elif type == "mixed":
            listOfSentences = self.preprocessor.getMixedSents()
        else:
            #Assume plain
            listOfSentences = self.preprocessor.gettokenizeSents()

        return 1

    def ngramExtraction(self, type, argString, preprocessReq):
        status, n, freq, filePOS, ngrams_file = self.ngramArgumentCheck(argString, type)
        if not status:
            # Error in argument.
            return

        # Handle preprocessing requests and exit
        if preprocessReq:
            self.preprocessReqHandle(type, filePOS)
            return 1

        if type == "plain":
            listOfSentences = self.preprocessor.gettokenizeSents()
        elif type == "POS":
            listOfSentences = self.preprocessor.getPOStagged(filePOS)
        elif type == "lemma":
            listOfSentences = self.preprocessor.getLemmatizedSents()
        elif type == "mixed":
            listOfSentences = self.preprocessor.getMixedSents()
        else:
            #Assume plain
            listOfSentences = self.preprocessor.gettokenizeSents()



        if ngrams_file: # read in a predefined list of ngrams
            with open(ngrams_file) as f:
                ngrams_str = f.read()
            ngrams_dict = ast.literal_eval(ngrams_str)
            #print(ngrams_dict)
            finNgram = ngrams_dict
            numberOfFeatures = len(finNgram)
        else:
            print("Building ngrams...")
            finNgram, numberOfFeatures = self.preprocessor.prep_servs.buildNgrams(
                                   n, freq, listOfSentences, pad_sentences=pad_sentences,
                                   ngram_type=type)



        print("Ngrams built.")

        if numberOfFeatures == 0:
            print("Cut-off too high, no ngrams passed it.")
            return []



        print(len(finNgram), numberOfFeatures)

        ngramFeatures = sparse.lil_matrix((len(listOfSentences), numberOfFeatures))

        print("Extracting ngram feats.")

        for i in range(len(listOfSentences)):
            ngramsVocab = Counter(ngrams(listOfSentences[i], n))
            lenSent = len(ngramsVocab)

            for ngramEntry in ngramsVocab:
                ## Keys
                ngramIndex = finNgram.get(ngramEntry, -1)
                if ngramIndex >= 0:
                    ngramFeatures[i, ngramIndex] = round((float(ngramsVocab[ngramEntry]) / lenSent), 2)

        print("Finished ngram features.")
        ngramLength = "Ngram feature vector length: " + str(numberOfFeatures)
        print(ngramLength)

        return ngramFeatures

    @featid(4)
    def ngramBagOfWords(self, argString, preprocessReq=0):
        '''
        Extracts n-gram bag of words features.
        '''
        return self.ngramExtraction("plain", argString, preprocessReq)

    @featid(5)
    def ngramBagOfPOS(self, argString, preprocessReq=0):
        '''
        Extracts n-gram POS bag of words features.
        '''
        return self.ngramExtraction("POS", argString, preprocessReq)

    @featid(6)
    def ngramBagOfMixedWords(self, argString, preprocessReq=0):
        '''
        Extracts n-gram mixed bag of words features.
        '''
        return self.ngramExtraction("mixed", argString, preprocessReq)

    @featid(7)
    def ngramBagOfLemmas(self, argString, preprocessReq=0):
        '''
        Extracts n-gram lemmatized bag of words features.
        '''
        return self.ngramExtraction("lemma", argString, preprocessReq)



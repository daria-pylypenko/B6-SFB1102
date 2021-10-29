'''
Created on Aug 23, 2016

@author: admin
'''
import scipy
import numpy as np
import sklearn
import random
from sklearn import model_selection
import os
import pickle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score


class Classifier(object):
    '''
    classdocs
    '''
        
    X = []
    y = []
       
    Xtrain = []
    ytrain = []
       
    Xtest = []
    ytest = []
       
    splitPercent = 0.3
    n_foldCV = 0
    threadCount = 1
    model = 0

    classifierName = ''
    
    def __init__(self, dataX, datay, threads=1, nCrossValid=1,
                             modelInput="", modelOutput = "",
                            train_size=0, val_size=0, random_state=None):
        self.X = dataX
        self.y = datay
        self.threadCount = threads
        self.n_foldCV = nCrossValid
        self.modelInput = modelInput
        self.modelOutput = modelOutput
        self.train_size = train_size
        self.val_size = val_size
        self.random_state = random_state

    def shuffle(self):
        self.X, self.y = sklearn.utils.shuffle(self.X, self.y)
        

    def splitTrainTest(self):
        #self.Xtrain, self.Xtest, self.ytrain, self.ytest = model_selection.train_test_split(self.X, self.y,
        #                                                                                    test_size=self.splitPercent,
        #                                                                                    random_state=0)

        # try random vectors
        #self.X = np.random.rand(self.X.shape[0], self.X.shape[1])

        # Use a predefined train-test boundary (train includes val too)
        # len_train + len_val
        train_test_boundary = self.train_size + self.val_size
        #train_test_boundary = 35846
        #train_test_boundary = 35847
        #train_test_boundary = 107541
        self.Xtrain = self.X[:train_test_boundary]
        self.Xtest = self.X[train_test_boundary:]
        self.ytrain = self.y[:train_test_boundary]
        self.ytest = self.y[train_test_boundary:]

        print("Unique labels")
        print(len(set(self.ytest)))

    def predict(self):
        return self.model.predict(self.Xtest)

    def evaluate(self):
        y_pred = self.predict()
        if len(set(self.ytest)) > 2:
            average = "micro"
        else:
            average = "binary"
        return accuracy_score(self.ytest, y_pred), precision_score(self.ytest, y_pred, average=average),\
               recall_score(self.ytest, y_pred, average=average), f1_score(self.ytest, y_pred, average=average)
        
    def runClassifier(self):
        """ Run the provided classifier."""
        acc = []; pre = []; rec = []; fsc = []

        print("Number of CV folds:", self.n_foldCV)
        for i in range(self.n_foldCV):
            #self.shuffle()
            self.splitTrainTest()
            self.train()
            accu, prec, reca, fsco = self.evaluate()
            acc.append(accu)
            pre.append(prec)
            rec.append(reca)
            fsc.append(fsco)

        classifReport = 'Average Accuracy: ' + str(np.mean(acc))
        classifReport += '\nAverage Precision: ' + str(np.mean(pre))
        classifReport += '\nAverage Recall: ' + str(np.mean(rec))
        classifReport += '\nAverage F-score: ' + str(np.mean(fsc))

        return classifReport

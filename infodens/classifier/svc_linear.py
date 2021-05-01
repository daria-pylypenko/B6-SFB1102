'''
Created on Aug 23, 2016

@author: admin
'''

from infodens.classifier.classifier import Classifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import numpy as np

import time


class SVC_linear(Classifier):
    '''
    classdocs
    '''

    classifierName = 'Support Vector Machine'
    C = np.logspace(-5.0, 5.0, num=10, endpoint=True, base=2)

    def train(self):

        tuned_parameters = [{'C': self.C}]

        print ('SVM Optimizing. This will take a while')
        start_time = time.time()
        #clf = GridSearchCV(LinearSVC(), tuned_parameters,
        #                   n_jobs=self.threadCount, cv=5)

        # use a single validation set
        # make a split index: 0's for val, -1's for train
        len_train_plus_val = 35846
        train_val_idx = np.zeros(len_train_plus_val)  # Xtrain is train + val
        # length of the train set:
        train_val_boundary = 29520
        train_val_idx[:train_val_boundary] = -1

        clf = GridSearchCV(LinearSVC(), tuned_parameters,
                           n_jobs=self.threadCount,
                           cv = PredefinedSplit(train_val_idx))

        clf.fit(self.Xtrain, self.ytrain)
        print('Done with Optimizing. it took ', time.time() -
              start_time, ' seconds')

        self.model = clf.best_estimator_

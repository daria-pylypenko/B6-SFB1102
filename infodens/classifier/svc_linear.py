'''
Created on Aug 23, 2016

@author: admin
'''

from infodens.classifier.classifier import Classifier
from joblib import dump, load
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

        #model_file = 'saved_models/direct_multi_model.joblib'
        #model_file = 'saved_models/direct_parallel_en_model.joblib'

        #np.random.seed(self.random_state)

        if self.modelInput: # load model from file
            self.model = load(self.modelInput)
        else:

            tuned_parameters = [{'C': self.C}]

            print ('SVM Optimizing. This will take a while')
            start_time = time.time()
            #clf = GridSearchCV(LinearSVC(), tuned_parameters,
            #                   n_jobs=self.threadCount, cv=5)

            # use a single validation set
            # make a split index: 0's for val, -1's for train
            #len_train_plus_val = 35846
            #len_train_plus_val = 35847
            #len_train_plus_val = 107541
            train_val_idx = np.zeros(self.Xtrain.shape[0])  # Xtrain is (train + val) x feature_size
            # length of the train set:
            train_val_boundary = self.train_size
            #train_val_boundary = 29520
            #train_val_boundary = 88560
            train_val_idx[:train_val_boundary] = -1


            print("Random state:", self.random_state)
            clf = GridSearchCV(LinearSVC(verbose=True, random_state=self.random_state), #fit_intercept=False),  # TODO: make this a feature (fit_intercept)
                               tuned_parameters,
                               n_jobs=6,
                               cv = PredefinedSplit(train_val_idx))

            clf.fit(self.Xtrain, self.ytrain)
            print('Done with Optimizing. it took ', time.time() -
                  start_time, ' seconds')

            self.model = clf.best_estimator_

            # save the model:
            if self.modelOutput:
                dump(self.model, self.modelOutput)
            #dump(self.model, 'saved_models/mono_el_en_downsampled_model.joblib')

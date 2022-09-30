#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import csv, os, re, sys, codecs
import numpy as np
import matplotlib.pyplot as plt
import joblib, statistics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter


import gensim.downloader as api

from eda import make_confusion_matrix

class data_classification():

    def __init__(self, path=None, clf_opt='lr', no_of_selected_features='all', label_path = "train_labels.csv", embedding = False):
        self.path = path
        self.label_path = label_path
        self.clf_opt = clf_opt
        self.no_of_selected_features = no_of_selected_features
        if self.no_of_selected_features is not None:
            self.no_of_selected_features = int(self.no_of_selected_features)
        self.embedding = embedding
        
        # Selection of classifiers

    def classification_pipeline(self):

        # Logistic Regression
        if self.clf_opt == 'lr':
            print('\n\t### Training Logistic Regression Classifier ### \n')
            clf = LogisticRegression(solver='liblinear', class_weight='balanced')
            clf_parameters = {
                'clf__random_state': (0, 10),
            }

        # Multinomial Naive Bayes
        elif self.clf_opt == 'nb':
            print('\n\t### Training Multinomial Naive Bayes Classifier ### \n')
            clf = MultinomialNB(fit_prior=True, class_prior=None)
            clf_parameters = {
                'clf__alpha': (0, 1),
            }

        # Random Forest
        elif self.clf_opt == 'rf':
            print('\n\t ### Training Random Forest Classifier ### \n')
            clf = RandomForestClassifier(max_features=None, class_weight='balanced')
            clf_parameters = {
                'clf__criterion': ('entropy', 'gini'),
                'clf__n_estimators': (30, 50, 100),
                'clf__max_depth': (10, 20, 30, 50, 100),
            }

        # Support Vector Machine
        elif self.clf_opt == 'svm':
            print('\n\t### Training SVM Classifier ### \n')
            clf = svm.SVC(class_weight='balanced', probability=True)
            clf_parameters = {
                'clf__C': (0.1, 1, 100),
                'clf__kernel':('linear','rbf'),
        }

        else:
            print('Select a valid classifier \n')
            sys.exit(0)
        return clf, clf_parameters

    # Load the data

    def get_data(self):

        data = pd.read_csv(self.path)
        labels = pd.read_csv(self.label_path).encoded_label
        # Training and Test Split
        trn_data, tst_data, trn_cat, tst_cat = train_test_split(data, labels, test_size=0.10, random_state=42,stratify=labels)

        return np.asarray(trn_data), np.asarray(tst_data), trn_cat, tst_cat

    # Classification using the Gold Standard after creating it from the raw text
    def classification(self):
        print("Starting Classification on ", self.path)
        # Get the data
        trn_data, tst_data, trn_cat, tst_cat = self.get_data()


        # splitting into train and val for evaluating with grid search and finding the best parameters
        # this will tell if the model is reliable for further experimentation
        average_confidence_scores = []

        skf = StratifiedKFold(n_splits=5)
        for train_index, test_index in skf.split(trn_data, trn_cat):

            X_train, y_train = trn_data[train_index], trn_cat.iloc[train_index]
            X_val, y_val = trn_data[test_index], trn_cat.iloc[test_index]
            

            clf, clf_parameters = self.classification_pipeline()
            
            if self.embedding:
                pipeline = Pipeline([
                ('clf', clf), ])
            else:    
                pipeline = Pipeline([
                    ('feature_selection', SelectKBest(chi2, k=self.no_of_selected_features) ),  # k=1000 is recommended
                    ('clf', clf), ])
            
            grid = GridSearchCV(pipeline, clf_parameters, scoring='f1_micro', cv=5)
            grid.fit(X_train, y_train)
            clf = grid.best_estimator_

            predicted = clf.predict(X_val)

            predicted_probability = np.max(clf.predict_proba(X_val), axis = 1)

            average_confidence_scores.append(round(statistics.mean(predicted_probability) - statistics.variance(predicted_probability), 3))

        average_confidence_scores = np.asarray(average_confidence_scores).mean()
        print('\n The average confidence score of the all classifier model configurations: \t' + str(average_confidence_scores) + '\n')

        # Evaluation of the latest model
        
        print('\n *************** Confusion Matrix ***************  \n')
        make_confusion_matrix(confusion_matrix(y_val, predicted))
        print(confusion_matrix(y_val, predicted))

        print('\n *************** Classification Report ***************  \n')
        print(classification_report(y_val, predicted))

        print('\n ***************  Scores on Training Data  *************** \n ')

        # Experiments on Given Test Data during Test Phase
        if average_confidence_scores > 0.50:

            print('\n ***** Classifying Test Data ***** \n')

            clf, clf_parameters = self.classification_pipeline()
           
            if self.embedding:
                pipeline = Pipeline([
                ('clf', clf), ])
            else:    
                pipeline = Pipeline([
                    ('feature_selection', SelectKBest(chi2, k=self.no_of_selected_features) ),  # k=1000 is recommended
                    ('clf', clf), ])
            

            grid = GridSearchCV(pipeline, clf_parameters, scoring='f1_micro', cv=5)

            grid.fit(trn_data,trn_cat)
            clf = grid.best_estimator_
            predicted = clf.predict(tst_data)

            print('\n ***************  Scores on Test Data  *************** \n ')
            
            report_dict = classification_report(tst_cat, predicted, output_dict=True)
            print(report_dict)
            print(classification_report(tst_cat, predicted))
            print(grid.best_params_)

            return report_dict
            
            
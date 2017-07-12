#
# General central framework to run stacked model to predict survival on the 
# Titanic. 
#
# Author: Charlie Bonfield
# Last Modified: July 2017

## Import statements 
# General 
import re
import sklearn 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from titanic_preprocessing import Useful_Preprocessing # conglomeration of stuff 
                                                       # used for preprocessing

# Base Models (assorted classification algorithms, may or may not use)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier

# Second Layer Model
import xgboost as xgb

# Helper function for all sklearn classifiers. 
class Sklearn_Helper(object):
    def __init__(self, classifier, seed=0, params=None):
        params['random_state'] = seed
        self.classifier = classifier(**params)

    def train(self, x_train, y_train):
        self.classifier.fit(x_train, y_train)

    def predict(self, x):
        return self.classifier.predict(x)
    
    def fit(self,x,y):
        return self.classifier.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.classifier.fit(x,y).feature_importances_)
        
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))

# Load in data. 
training_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Impute missing 'Fare' values with median.
training_data['Fare'] = training_data['Fare'].fillna(training_data['Fare'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
    
# Combine data for preprocessing (should not be an issue, as most of this is 
# just casting categorical features as numbers and dropping things we do not
# wish to use).
pp = Useful_Preprocessing()
combined = pd.concat([training_data, test_data])
combined = pp.transform_all(combined)
combined = pp.impute_ages(combined) # may not be permissible to do this on the
                                    # training/test set simultaneously - need
                                    # to read paper on MICE to be sure.
                                    
# Split back out into training/test sets. 
training_data = combined[:891]
test_data = combined[891:].drop('Survived', axis=1)

# Standardize age/fare features. 
scaler = preprocessing.StandardScaler()
select = 'Age Fare'.split()
training_data[select] = scaler.fit_transform(training_data[select])
test_data[select] = scaler.transform(test_data[select])
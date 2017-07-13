#
# General central framework to run stacked model to predict survival on the 
# Titanic. 
#
# Author: Charlie Bonfield
# Last Modified: July 2017

## Import statements 
# General 
#import re
import scipy 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from titanic_preprocessing import Useful_Preprocessing # conglomeration of stuff 
                                                       # used for preprocessing

# Base Models (assorted classification algorithms, may or may not use all of 
# these)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.grid_search import RandomizedSearchCV # old sklearn
#from sklearn.model_selection import RandomizedSearchCV # new sklearn
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

# Perform hyperparameter tuning for given training set (will do this implicitly
# for every fold).  
def hyperparameter_tuning(classifier, param_dist, n_iterations, X, y):
    clf = classifier()
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iterations, n_jobs=-1)
    random_search.fit(X, y)
    best_params = random_search.best_params_
    return best_params

# Generate out-of-fold predictions for training set. For each fold, generate
# a prediction for the test set, then return the average of those predictions
# for out test set "meta feature". 
def get_out_of_fold_predictions(classifier, param_dist, kf, n_folds, x_train, 
                                y_train, n_train, x_test, n_test, seed):
    oof_train = np.zeros((n_train,))
    oof_test = np.zeros((n_test,))
    oof_test_full = np.empty((n_folds, n_test))

    # Iterate over sets of training/test indices corresponding to each fold.
    for i, (train_indices, test_indices) in enumerate(kf):
        #print(train_index)
        #print(test_index)
        x_tr = x_train[train_indices]
        y_tr = y_train[train_indices]
        x_te = x_train[test_indices]
        
        best_params = hyperparameter_tuning(classifier, param_dist, 50, x_tr, y_tr)
        clf = Sklearn_Helper(classifier, seed=seed, params=best_params)

        clf.train(x_tr, y_tr)

        oof_train[test_indices] = clf.predict(x_te)
        oof_test_full[i, :] = clf.predict(x_test)
    
    # Take the average prediction of all folds to be our "meta feature" for
    # the actual test set.
    oof_test[:] = oof_test_full.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# Load in data. 
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Impute missing 'Fare' values with median.
train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())
    
# Combine data for preprocessing (should not be an issue, as most of this is 
# just casting categorical features as numbers and dropping things we do not
# wish to use).
pp = Useful_Preprocessing()
combined = pd.concat([train_data, test_data])
combined = pp.transform_all(combined)
combined = pp.impute_ages(combined) # may not be permissible to do this on the
                                    # training/test set simultaneously - need
                                    # to read paper on MICE to be sure.
                                    
# Split back out into training/test sets. 
train_data = combined[:891]
test_data = combined[891:].drop('Survived', axis=1)

# Standardize age/fare features. 
scaler = preprocessing.StandardScaler()
select = 'Age Fare'.split()
train_data[select] = scaler.fit_transform(train_data[select])
test_data[select] = scaler.transform(test_data[select])

# Prepare for stacking (these parameters will be needed to generate out-of-fold
# predictions). 
n_train = train_data.shape[0]
n_test = test_data.shape[0]
SEED = 42   
NFOLDS = 5    # set the number of folds for out-of-fold prediction
kf = KFold(n_train, n_folds= NFOLDS, random_state=SEED)

# Split out target feature for training set, rename train/test data for convenience.
y_train = train_data['Survived'].ravel()
train_data = train_data.drop(['Survived'], axis=1)
x_train = train_data.values 
x_test = test_data.values

# Provide set of parameter distributions to be searched by RandomSearchCV
# for each classifer (needed for tuning, can be customized).
#
# SAMPLE: param_dist = {'C': scipy.stats.uniform(0.1, 1000), 
#                       'gamma': scipy.stats.uniform(.001, 1.0),
#                       'kernel': ['rbf'], 'class_weight':['balanced', None]}
#
svc_dist = {'C': scipy.stats.uniform(0.1,1000), 
            'gamma': scipy.stats.uniform(.001,1.0),
            'kernel': ['rbf'], 'class_weight':['balanced', None]} 
ada_dist = {'n_estimators': scipy.stats.uniform(1,100),
            'learning_rate': scipy.stats.uniform(.001,1.0)}
rf_dist = {'n_estimators': scipy.stats.uniform(1,100), 'warm_start': [True],
           'max_depth': scipy.stats.uniform(2.0,6.0), 
           'min_samples_leaf': scipy.stats.uniform(1.0,3.0)}
gb_dist = {'n_estimators': scipy.stats.uniform(1,100), 'warm_start': [True],
           'max_depth': scipy.stats.uniform(2.0,6.0), 
           'min_samples_leaf': scipy.stats.uniform(1.0,3.0)}
et_dist = {'n_estimators': scipy.stats.uniform(1,100), 'warm_start': [True],
           'max_depth': scipy.stats.uniform(2.0,6.0), 
           'min_samples_leaf': scipy.stats.uniform(1.0,3.0)}

# Generate first-level predictions. 
# Arguments: (classifer, param_dist, kf, n_folds, x_train, y_train, n_train, 
#             x_test, n_test, seed)
print('Generating first-level predictions:')
svc_fl_train, svc_fl_test = get_out_of_fold_predictions(SVC,svc_dist,kf,NFOLDS,
                                                        x_train, y_train, n_train,
                                                        x_test, n_test, SEED)
ada_fl_train, ada_fl_test = get_out_of_fold_predictions(AdaBoostClassifier,
                                                        ada_dist, kf, NFOLDS,
                                                        x_train, y_train, n_train,
                                                        x_test, n_test, SEED)
rf_fl_train, rf_fl_test = get_out_of_fold_predictions(RandomForestClassifier,
                                                      rf_dist,kf,NFOLDS,x_train, 
                                                      y_train, n_train, x_test,
                                                      n_test, SEED)
gb_fl_train, gb_fl_test = get_out_of_fold_predictions(GradientBoostingClassifier,
                                                      gb_dist,kf,NFOLDS,x_train, 
                                                      y_train, n_train, x_test,
                                                      n_test, SEED)
et_fl_train, et_fl_test = get_out_of_fold_predictions(ExtraTreesClassifier,
                                                      et_dist,kf,NFOLDS,x_train, 
                                                      y_train, n_train, x_test, 
                                                      n_test, SEED)

# Save results, will be fed into second level. 
x_train_meta = np.concatenate((svc_fl_train,ada_fl_train,rf_fl_train,gb_fl_train,
                               et_fl_train), axis=1)
x_test_meta = np.concatenate((svc_fl_test,ada_fl_test,rf_fl_test,gb_fl_test,
                              et_fl_test), axis=1)
x_train_meta.dump('meta_train.dat')
x_test_meta.dump('meta_test.dat')
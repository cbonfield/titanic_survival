#
# Set of functions used for preprocessing Titanic data. 
#
# Author: Charlie Bonfield
# Last Modified: July 2017

# Import statements
import fancyimpute
import numpy as np
import pandas as pd
from sklearn import preprocessing

class Useful_Preprocessing(object):
    
    # Code performs three separate tasks:
    #   (1). Pull out the first letter of the cabin feature.
    #          Code taken from: https://www.kaggle.com/jeffd23/titanic/scikit-learn-ml-from-start-to-finish
    #   (2). Add column which is binary variable that pertains
    #        to whether the cabin feature is known or not.
    #        (This may be relevant for Pclass = 1).
    #   (3). Recasts cabin feature as number.
    def simplify_cabins(self, data):
        data.Cabin = data.Cabin.fillna('N')
        data.Cabin = data.Cabin.apply(lambda x: x[0])

        cabin_mapping = {'N': 0, 'A': 1, 'B': 1, 'C': 1, 'D': 1, 'E': 1,
                         'F': 1, 'G': 1, 'T': 1}
        data['Cabin_Known'] = data.Cabin.map(cabin_mapping)

        le = preprocessing.LabelEncoder().fit(data.Cabin)
        data.Cabin = le.transform(data.Cabin)

        return data

    # Recast sex as numerical feature.
    def simplify_sex(self, data):
        sex_mapping = {'male': 0, 'female': 1}
        data.Sex = data.Sex.map(sex_mapping).astype(int)

        return data
    
    # Recast port of departure as numerical feature.
    def simplify_embark(self, data):
        # Two missing values, assign the most common port of departure.
        data.Embarked = data.Embarked.fillna('S')

        le = preprocessing.LabelEncoder().fit(data.Embarked)
        data.Embarked = le.transform(data.Embarked)

        return data

    # Extract title from names, then assign to one of five ordinal classes.
    # Function based on code from: https://www.kaggle.com/startupsci/titanic/titanic-data-science-solutions
    def add_title(self, data):
        data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        data.Title = data.Title.replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                                         'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        data.Title = data.Title.replace('Mlle', 'Miss')
        data.Title = data.Title.replace('Ms', 'Miss')
        data.Title = data.Title.replace('Mme', 'Mrs')

        # Map from strings to ordinal variables.
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

        data.Title = data.Title.map(title_mapping)
        data.Title = data.Title.fillna(0)

        return data

    # Drop all unwanted features (name, ticket).
    def drop_features(self, data):
        return data.drop(['Name', 'Ticket'], axis=1)
    
    # Perform all feature transformations.
    def transform_all(self, data):
        data = self.simplify_cabins(data)
        data = self.simplify_sex(data)
        data = self.simplify_embark(data)
        data = self.add_title(data)
        data = self.drop_features(data)
        return data
    
    # Impute missing ages using MICE.
    def impute_ages(self, data):
        drop_survived = data.drop(['Survived'], axis=1)
        column_titles = list(drop_survived)
        mice_results = fancyimpute.MICE().complete(np.array(drop_survived))
        results = pd.DataFrame(mice_results, columns=column_titles)
        results['Survived'] = list(data['Survived'])
        return results
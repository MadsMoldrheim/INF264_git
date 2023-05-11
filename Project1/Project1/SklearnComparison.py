# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 17:28:35 2022

@author: madsm
"""
from sklearn import model_selection, tree
import numpy as np
from datetime import datetime
startTime = datetime.now()

def dataSets(seed, file):
    """
    Importing data as both float and string
    storing features in a matrix of floats
    and labels in an array of strings
    
    Parameters
    ----------
    seed : integer, seed for splitting of datasets to ensure repeatability
    
    Returns
    -------
    matrixes of floats, datasets for training, validation and testing
    + a dictionary for convertion from integer-lables to string-labels
    """
    DataFloat = np.genfromtxt(file, delimiter=',')
    DataString = np.genfromtxt(file, dtype=str, delimiter=',')

    X = DataFloat[:, 0:-1]
    Y = DataString[:, -1]
    
    #Convert y labels from strings to integers
    #and create a dictionary to allow for convertion from integer to string later
    Y_dict = dict(zip(set(Y), range(len(Y))))
    Y_int = np.array([Y_dict[x] for x in Y])

    #Deviding data into training, validation and test data
    X_train, X_val_test, Y_train, Y_val_test = model_selection.train_test_split(
        X, Y_int, 
        test_size=0.3, shuffle=True, random_state=seed
    )

    # Shuffle and split the data into validation and test sets with a ratio of 0.5/0.5:
    X_val, X_test, Y_val, Y_test = model_selection.train_test_split(
        X_val_test, Y_val_test, 
        test_size=0.5, shuffle=True, random_state=seed
    )
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_dict


file = "magic04.data"
seed = 123
X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_dict = dataSets(seed, file)


#_-_-_-_-_-_-_Sklearn_-_-_-_-_-_-_#
#Using the already existing tools of the sklearn package
#and seeing how they compare to the model in this code

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)
score = clf.score(X_test, Y_test)
print("\nSklearn test accuracy:\t", score)
print("\nTime spent:\t\t", datetime.now() - startTime)
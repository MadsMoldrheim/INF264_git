Author:
    Mads Moldrheim

Date:
    23.09.2022
    
Files:
    DecisionTree_wPruning.py
    sklearnComparison.py
    magic04.data

This code is the mandatory assignment #1 in the machine learning course INF264 at UiB

The code takes a dataset of features and labels and builds a decision tree.
It can also do reduced-error pruning. To prune, simply set the variable pruning = True

In it's current state, the code will calculate the accuracy of the tree with the validation data.
By using the function predict() one can predict a label for a single data point.
The function accuracy() will return the accuracy of the tree on a complete dataset with features and labels

To run the code, simply make sure that the file "magic04.data"
is in the same folder as the code DecisionTree_wPruning.py
The sklearnComparison.py is just using the decision tree clasifier
from the sklearn library to compare it to my own code
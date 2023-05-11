import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets, model_selection


class Node:
    def __init__(self, feature, value, leftChild, rightChild, isLeaf = False, label = None):
        self.feature = feature
        self.value = value
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.isLeaf = isLeaf
        self.label = label

def dataSets():
    #Importing data as both float and string
    #storing features in a matrix of floats and labels in an array of strings
    DataFloat = np.genfromtxt('magic04.data', delimiter=',')
    DataString = np.genfromtxt('magic04.data', dtype=str, delimiter=',')

    X = DataFloat[:, 0:-1]
    Y = DataString[:, -1]
    Y_dict = dict(zip(set(Y), range(len(Y))))
    Y_int = np.array([Y_dict[x] for x in Y])

    #Deviding data into training, validation and test data
    seed = 123
    X_train, X_val_test, Y_train, Y_val_test = model_selection.train_test_split(
        X, Y_int, 
        test_size=0.3, shuffle=True, random_state=seed
    )

    # Shuffle and split the data into validation and test sets with a ratio of 0.5/0.5:
    X_val, X_test, Y_val, Y_test = model_selection.train_test_split(
        X_val_test, Y_val_test, 
        test_size=0.5, shuffle=True, random_state=seed
    )
    
    Xy = np.concatenate([X_train, np.vstack(Y_train)], axis=1)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test, Xy

def learn(Xy, impurity_measure="entropy"):

    X = Xy[:, :-1]
    y = Xy[:, -1]

    yvalue,ycounts = np.unique(y, return_counts=True)
    xvalue,xcounts = np.unique(X, return_counts=True)

    #If all data points have the same label
    #return a leaf with that label
    if len(ycounts) == 1:
        #Return leaf with value
        return Node(None,
             None,
             None,
             None,
             True,
             yvalue)
        print("if")

    #Else if all data points have identical feature values
    #return a leaf with the most common label
    elif len(xcounts) == 1:
        label = yvalue[np.where(ycounts==max(ycounts))]
        return Node(None,
             None,
             None,
             None,
             True,
             label)
        print("elif")
    
    #Else
    #choose a feature that maximizes the information gain
    #split the data based on the value of the feature and add a branch
    #for each subset of data
    #for each branch
    else:
        best = best_split(Xy, impurity_measure)
        
        leftChild = learn(best["leftChild"], impurity_measure)        
        rightChild = learn(best["rightChild"], impurity_measure)
        
        return Node(best["feature"],
                    best["value"],
                    leftChild,
                    rightChild)

def best_split(Xy, impurity_measure):
    #Number of rows and columns in Xy
    Xy_rows, Xy_cols = Xy.shape
    X_cols = Xy_cols - 1
    
    #for every column of features in Xy, calculate witch feature to split
    #based on a split on the mean value
    best_split = {"ig":0, "value":None, "feature":None, "leftChild":None, "rightChild":None}
    for col in range(X_cols):
        #Check that there are multiple unique values in the feature
        #If there is only one value in the entire feature,
        #there is nowhere to split
        if len(np.unique(Xy[:, col])) > 1:
            avg = np.average(Xy[:,col])
            
            #X sorted by column
            #Xy_sort = Xy[Xy[:, col].argsort()]
            
            #Splits the matrix into two matrixes based on the split value
            leftXy = np.array([row for row in Xy if row[col] <= avg])
            rightXy = np.array([row for row in Xy if row[col] > avg])
            
            leftXy_rows, leftXy_cols = leftXy.shape
            rightXy_rows, rightXy_cols = rightXy.shape
    
            #Calculate information gain
            ig = impurity(Xy[:,-1], impurity_measure) - ((leftXy_rows)/(Xy_rows) * impurity(leftXy[:, -1], impurity_measure) + rightXy_rows/Xy_rows * impurity(rightXy[:, -1], impurity_measure))
            #If the information gain is the best yet, store it with features as the best split
            if ig > best_split["ig"]:
                best_split = {"ig":ig, "value":avg, "feature":col, "leftChild": leftXy, "rightChild":rightXy}
                
    return best_split

def impurity(X, impurity_measure = "entropy"):
    
    value,counts = np.unique(X, return_counts=True)
    
    ProbX = counts / counts.sum()


    if impurity_measure == "entropy":
        #Calculate entropy
        return -np.multiply(ProbX, np.log2(ProbX)).sum()
        
    elif impurity_measure == "gini":
        #Calculate Gini index
        return np.multiply(ProbX, 1-ProbX).sum()
    else:
        print("Invalid impurity measure")
        
def predict(x, tree):
    print(tree)
    if tree.isLeaf:
        return tree.label
    else:
        col = tree.feature
        value = tree.value
        if x[col] <= value:
            #Go to left child
            return predict(x=x, tree=tree.leftChild)
        elif x[col] > value:
            #Go to right child
            return predict(x=x, tree=tree.rightChild)
        
def accuracy(X, y, tree):
    prediction = (np.array([int(predict(x, tree)) for x in X])).flatten()
    correct = prediction == y
    accuracy = correct.sum()/len(correct)
    return accuracy
        
#_-_-_-_-_-_-_Main code_-_-_-_-_-_-_#

#Import data and split it into training, validation and test sets
X_train, X_val, X_test, Y_train, Y_val, Y_test, Xy = dataSets()

root = learn(Xy)

trainAcc = accuracy(X_train, Y_train, root)
valAcc = accuracy(X_val, Y_val, root)
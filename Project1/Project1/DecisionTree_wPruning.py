import numpy as np
from sklearn import model_selection, tree
from datetime import datetime
startTime = datetime.now()

#Initiating node class
class Node:
    def __init__(self, feature, value, leftChild, rightChild, label, pruneLabels = None, isLeaf = False):
        self.feature = feature
        self.value = value
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.label = label
        self.pruneLabels = pruneLabels
        self.isLeaf = isLeaf
        
        
#_-_-_-_-_-_-_Functions_-_-_-_-_-_-_#
        
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

def learn(Xy, impurity_measure, Xy_prune = None, pruning=False):
    """
    Function for building the decision tree, called recursively
    Greedy ID3 algorithm

    Parameters
    ----------
    Xy                  : Matrix of floats, features and labels
    impurity_measure    : String, witch impurity measure to use for calculating information gain
    Xy_prune            : Matrix of floats, features and labels from the pruning set
    pruning             : Boolean, wehter to prune or not

    Returns
    -------
    Nodes to append to the root node
    
    """
    #Split the matrix into features and labels
    X = Xy[:, :-1]
    y = Xy[:, -1]
      
    #Get the unique values and their count
    yvalue,ycounts = np.unique(y, return_counts=True)
    xvalue,xcounts = np.unique(X, return_counts=True)
    
    #If pruning, get the unique values of the labels from the pruning data and their counts
    #This data will be used for calculating the change in accuracy when deciding to prune or not
    if pruning:
        pruneUnique = np.unique(np.array([row[-1] for row in Xy_prune]), return_counts=True)
    else:
        pruneUnique = None
        
    #If all data points have the same label
    #return a leaf with that label
    if len(ycounts) == 1:
        #Return leaf with value
        return Node(None,
             None,
             None,
             None,
             yvalue,
             pruneUnique,
             True)

    #Else if all data points have identical feature values
    #return a leaf with the most common label
    elif len(xcounts) == 1:
        label = yvalue[np.where(ycounts==max(ycounts))]
        return Node(None,
             None,
             None,
             None,
             label,
             pruneUnique,
             True)
    
    #Else
    #choose a feature that maximizes the information gain
    #split the data based on the value of the feature and add a branch
    #for each subset of data
    #for each branch
    else:
        #Call the function best_split() to get the best split
        best = best_split(Xy, impurity_measure, Xy_prune, pruning)
        
        #Split the dataset as chosen by best_split() and call learn() recursively
        leftChild = learn(best["leftChild"], impurity_measure, best["leftChildPrune"], pruning)        
        rightChild = learn(best["rightChild"], impurity_measure, best["rightChildPrune"], pruning)
        
        return Node(best["feature"],
                    best["value"],
                    leftChild,
                    rightChild,
                    best["label"],
                    pruneUnique)

def best_split(Xy, impurity_measure = "entropy", Xy_prune=None, pruning=False):
    """
    Calculate the best split among the features
    Each feature is split on the average value

    Parameters
    ----------
        
    Xy                  : Matrix of floats, features and labels
    impurity_measure    : String, witch impurity measure to use for calculating information gain
    Xy_prune            : Matrix of floats, features and labels from the pruning set
    pruning             : Boolean, wehter to prune or not

    Returns
    -------
    best_split : Dictionary, containing values to make the split in learn()

    """
    #Number of rows and columns in Xy
    Xy_rows, Xy_cols = Xy.shape
    
    #Number of columns in just the features X
    X_cols = Xy_cols - 1
    
    #Calculate majority label in case of pruning
    y = Xy[:, -1]
    yvalue,ycounts = np.unique(y, return_counts=True)
    label = yvalue[np.where(ycounts==max(ycounts))]
    
    #If there is a tie, give no label
    #In pruning this will result in not pruning the subtree
    if len(label) > 1:
        label = None
    
    #for every column of features in Xy, calculate witch feature to split
    #based on a split on the average value
    best_split = {"ig":0,
                  "value":None,
                  "feature":None,
                  "leftChild":None,
                  "rightChild":None,
                  "label":label,
                  "leftChildPrune":None,
                  "rightChildPrune":None}
    
    for col in range(X_cols):
        #Check that there are multiple unique values in the feature
        #If there is only one value in the entire feature,
        #there is nowhere to split
        if len(np.unique(Xy[:, col])) > 1:
            avg = np.average(Xy[:,col])
            
            #Splits the matrix into two matrixes based on the split value
            leftXy = np.array([row for row in Xy if row[col] <= avg])
            rightXy = np.array([row for row in Xy if row[col] > avg])
            
            #If pruning, split the dataset for pruning
            if pruning:
                leftXy_prune = np.array([row for row in Xy_prune if row[col] <= avg])
                rightXy_prune = np.array([row for row in Xy_prune if row[col] > avg])
            else:
                leftXy_prune, rightXy_prune = None, None
                
            #Get the dimensions of the new child sets
            #Used for calculating the collective impurity for the two datasets
            leftXy_rows, leftXy_cols = leftXy.shape
            rightXy_rows, rightXy_cols = rightXy.shape
    
            #Calculate information gain
            ig = impurity(Xy[:,-1], impurity_measure) - ((leftXy_rows)/(Xy_rows) * impurity(leftXy[:, -1], impurity_measure) + rightXy_rows/Xy_rows * impurity(rightXy[:, -1], impurity_measure))
            
            #If the information gain is the best yet, store it with features as the best split
            if ig > best_split["ig"]:
                best_split = {"ig":ig,
                              "value":avg,
                              "feature":col,
                              "leftChild": leftXy,
                              "rightChild":rightXy,
                              "label":label,
                              "leftChildPrune":leftXy_prune,
                              "rightChildPrune":rightXy_prune}
                
    return best_split

def impurity(X, impurity_measure = "entropy"):
    """
    Calculate impurity, either as entropy or gini

    Parameters
    ----------
    X                   : Array, data to calculate impurity for
    impurity_measure    : String, witch impurity measure to calculate

    Returns
    -------
    Float, the calculated impurity

    """
    #Get the unique values in X and their counts
    value,counts = np.unique(X, return_counts=True)
    
    #Create an array with the probabilities for each unique x value
    ProbX = counts / counts.sum()


    if impurity_measure == "entropy":
        #Calculate and return entropy
        return -np.multiply(ProbX, np.log2(ProbX)).sum()
        
    elif impurity_measure == "gini":
        #Calculate and return Gini index
        return np.multiply(ProbX, 1-ProbX).sum()
    else:
        print("Invalid impurity measure")
        
def predict(x, tree):
    """
    Predict a label for a given data point using a tree
    
    Parameters
    ----------
    x       : Array, data point to predict label for
    tree    : Class, the tree to use for prediction
    
    Returns
    -------
    Integer, predicted label

    """
    #Check if the node is a leaf node, return the label if it is
    if tree.isLeaf:
        return tree.label
    
    #Find witch feature the node splits on and follow that branch
    #With the child node, call the function recursively until a leaf node is found
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
    """
    Calculate the accuracy of the predicted labels

    Parameters
    ----------
    X       : Matrix of features, data to predict on
    y       : Array of labels, correct labels for the dataset X
    tree    : Class, tree to use for predictions

    Returns
    -------
    accuracy : Float, the accuracy of predictions

    """
    #Predict labels for each datapoint x in X and concoct them to in an array
    prediction = (np.array([int(predict(x, tree)) for x in X])).flatten()
    
    #Compare the predictions to the actual labels
    compare = prediction == y
    
    #Calculate the accuracy of the predictions using the comparison array
    accuracy = compare.sum()/len(compare)
    return accuracy    

def prune(tree):
    """
    Do one iteration of reduced-error pruning on the tree

    Parameters
    ----------
    tree : Class, tree to prune

    Returns
    -------
    None.

    """

    #When you hit a leaf node, do nothing
    if tree.isLeaf:
        return
    
    #Upon finding a node with two leaf nodes as children that also has both pruning data and a majority label, try to prune
    elif tree.leftChild.isLeaf and tree.rightChild.isLeaf and tree.pruneLabels is not None and tree.label is not None:
        
        #Get the value and counts for labels from the pruning set that has been directed to this node
        nodeValue, nodeCounts = tree.pruneLabels
        
        #Calculate the accuracy for pruning data on this nodes majority label
        accNode = nodeCounts[np.where(nodeValue==tree.label)]/sum(nodeCounts)
        
        #If there are no instances of pruning labels that mach the majority label,
        #the array will return empty and the accuracy is =0
        if len(accNode) < 1: accNode = 0
        
        #If there is no pruning data associated with the child nodes,
        #there is no data to support a pruning and we do not prune
        if tree.leftChild.pruneLabels is not None and tree.rightChild.pruneLabels is not None:
            
            #Get the value and counts for labels from the pruning set that has been directed to the child nodes
            leftLeafValue, leftLeafCounts = tree.leftChild.pruneLabels
            rightLeafValue, rightLeafCounts = tree.rightChild.pruneLabels
            
            #Calculate the collective accuracy in the two leaf nodes
            accLeafs = (leftLeafCounts[np.where(leftLeafValue==tree.leftChild.label)] + rightLeafCounts[np.where(rightLeafValue==tree.rightChild.label)])/sum(nodeCounts)
            
            #If there are no instances of pruning labels that mach the majority labels,
            #the array will return empty and the accuracy is =0
            if len(accLeafs) < 1: accLeafs = 0
            
            #If the node accuracy is equal or greater than the leaf accuracy,
            #prune the leafs and make the parent node into a leaf node
            if accNode >= accLeafs:
                tree.isLeaf = True
                tree.leftChild = None
                tree.rightChild = None
                return
            
    #Recursively traverse the tree
    prune(tree.leftChild)
    prune(tree.rightChild)
            
        
     
#_-_-_-_-_-_-_Main code_-_-_-_-_-_-_#

file = "magic04.data"
seed = 123
impurity_measure = "entropy"
pruning = True

print("Impurity measure:\t\t", impurity_measure)
print("Pruning:\t\t\t\t", pruning)

X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_dict = dataSets(seed, file)

if pruning:
    #Split data into train and pruning data
    X_train, X_prune, Y_train, Y_prune = model_selection.train_test_split(
        X_train, Y_train, 
        test_size=0.2, shuffle=True, random_state=seed
    )
    Xy = np.concatenate([X_train, np.vstack(Y_train)], axis=1)
    Xy_prune = np.concatenate([X_prune, np.vstack(Y_prune)], axis=1)
    
    #Build the tree
    root = learn(Xy, impurity_measure, Xy_prune, pruning)
    
    #Display accuracy before pruning
    print("\nPre pruning:")
    trainAcc = accuracy(X_train, Y_train, root)
    valAcc = accuracy(X_val, Y_val, root)
    print("Training accuracy:\t\t", trainAcc)
    print("Validation accuracy:\t", valAcc)
    
    #Create a while loop that iteratively prunes the tree as long as accuracy increases
    iterating = True
    prevAcc = valAcc
    while iterating:
        prune(root)
        currAcc = accuracy(X_val, Y_val, root)
        if not currAcc > prevAcc:
            iterating = False
        prevAcc = currAcc
        print("\nIterative pruning")
    
    #Display accuracy after pruning
    print("\nAfter pruning:")
    trainAcc = accuracy(X_train, Y_train, root)
    valAcc = accuracy(X_val, Y_val, root)
    print("Training accuracy:\t\t", trainAcc)
    print("Validation accuracy:\t", valAcc)
    
else:
    Xy = np.concatenate([X_train, np.vstack(Y_train)], axis=1)
    
    #Build the tree
    root = learn(Xy, impurity_measure)
    
    #Display the accuracy
    trainAcc = accuracy(X_train, Y_train, root)
    valAcc = accuracy(X_val, Y_val, root)
    print("\nTraining accuracy:\t\t", trainAcc)
    print("Validation accuracy:\t", valAcc)

#Print the time it has taken for the code to execute
print("\nTime spent:\t\t", datetime.now() - startTime)
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.metrics import plot_confusion_matrix
import random


class kNNclass:
    """
    Class for the kNN models
    Stores the model itself
    along with hyperparameters and accuracies
    """
    def __init__(self, model, k, weight, train_acc, val_acc):
        self.model = model
        self.k = k
        self.weight = weight
        self.train_acc = train_acc
        self.val_acc = val_acc
        
    def __str__(self):
        return "k-Nearest Neighbor classifier with parameters k=" + str(self.k) + " and weight: " + str(self.weight) + ". Validation accuracy is " + str(self.val_acc)
        
        
class DTCclass:
    """
    Class for the decision tree models
    Stores the model itself
    along with hyperparameters and accuracies
    """
    def __init__(self, model, impurity, min_samples_split, train_acc, val_acc):
        self.model = model
        self.impurity = impurity
        self.min_samples_split = min_samples_split
        self.train_acc = train_acc
        self.val_acc = val_acc
    
    def __str__(self):
        return "Decision tree classifier with parameters min_samples_split=" + str(self.min_samples_split) + " and impurity: " + str(self.impurity) + ". Validation accuracy is " + str(self.val_acc)
        
    
class MLPclass:
    """
    Class for the neural network models
    Stores the model itself
    along with hyperparameters and accuracies
    """
    def __init__(self, model, hidden_depth, nodes_per_layer, train_acc, val_acc):
        self.model = model
        self.hidden_depth = hidden_depth
        self.nodes_per_layer = nodes_per_layer
        self.train_acc = train_acc
        self.val_acc = val_acc
        
    def __str__(self):
        return "Neural network classifier with hidden layers: " + str(self.hidden_depth) + " and nodes per layer: " + str(self.nodes_per_layer) + ". Validation accuracy is " + str(self.val_acc)
                

def balanceData(X, y, seed):
    """
    Identifies over- or undersampled labels in the dataset and balances the data set
    
    Parameters
    ----------
    X       : Matrix, features
    y       : array, labels
    seed    : integer, seed for ensuring repeatability in respect to random effects
    
    Returns
    -------
    matrixes of balanced data set
    """
    Xy = np.concatenate([X, np.vstack(y)], axis=1)
    values, counts = np.unique(y, return_counts=True)
    average = sum(counts)/len(counts)
    
    #List to store data as balanced, undersampled or oversampled
    balanced = []
    undersampled = []
    oversampled = []
    
    #Loop through labels and find data to keep, oversample or undersample
    for i in range(len(counts)):
        frac = counts[i]/average
        
        if 0.5 <= frac <= 1.5:
            balanced.append([row for row in Xy if row[-1]==values[i]])
            
            
        elif frac < 0.5:
            undersampled.append([row for row in Xy if row[-1]==values[i]])
            
        elif frac > 1.5:
            oversampled.append([row for row in Xy if row[-1]==values[i]])
    
    
    balanced = np.vstack(balanced)
    
    #Calculate new average as a goal for oversampling/undersampling
    values, counts = np.unique(balanced[:, -1], return_counts=True)
    average = sum(counts)/len(counts)
    
    #Oversampling the undersampled labels until it matches the average
    for i in undersampled:
        array = []
        newPoints = int(average) - len(i)
        for _ in range(newPoints):
            array.append(random.choice(i))
        array.append(i)
        array = np.vstack(array)
        balanced = np.vstack((balanced, array))
        
    for i in oversampled:
        newPoints = len(i) - int(average)
        for _ in range(newPoints):
            i.pop(random.randrange(len(i)))
        array = []
        array = np.vstack(i)
        balanced = np.vstack((balanced, array))
    
    balanced = shuffle(balanced, random_state = seed)
    return balanced[:, :-1], balanced[:, -1]


def dataSets(seed, X, y, test_size, val_size):
    """
    Importing data as both float and string
    storing features in a matrix of floats
    and labels in an array of strings
    
    Parameters
    ----------
    seed        : integer, seed for splitting of datasets to ensure repeatability
    X           : Matrix, features
    y           : array, labels
    test_size   : float, fraction of data from X_val_test and y_val_test that becomes test data
    val_size    : float, fraction of data from X and y that becomes X_val_test and y_val_test
    
    Returns
    -------
    matrixes, datasets for training, validation and testing
    """
    #Dividing data into training, validation and test data
    X_train, X_val_test, y_train, y_val_test = model_selection.train_test_split(
       X, y, 
       test_size=test_size, shuffle=True, random_state=seed
    )

    X_val, X_test, y_val, y_test = model_selection.train_test_split(
       X_val_test, y_val_test, 
       test_size=val_size, shuffle=True, random_state=seed
       )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def weighted_acc(predictions, labels):
    """
    Calculates the average accuracy of predictions
    
    Parameters
    ----------
    predictions : array, predictions to calculate accuracy for
    labels      : array, treu labels to compare the predictions to
    
    Returns
    -------
    average accuracy
    """
    
    #Get unique labels and their count from the the array labels
    value,counts = np.unique(labels, return_counts=True)
    
    #Create a dictionary of labels as keys and set their value as 0
    d = {}
    for i in value:
        d[i] = 0
        
    #Loop through the predictions and sum up the amount of correct predictions for each unique label
    for p, l in zip(predictions, labels):
        d[l] += p==l

    SUM = 0
    for i in d:
        SUM += float(d[i]/counts[np.where(value==i)])
        
    return SUM/value.size

def kNearestNeighbors(kk, weight):
    """
    Creates and stores kNN-models with various hyperparamers
    
    Parameters
    ----------
    kk      : array, values for k
    weight  : array, how to weigh neighbors
    
    Returns
    -------
    kNN_models : nested list of class objects, containing model with model data
    """

    kNN_models = []

    for w in weight:
        array = []
        for k in kk:
            kNN = KNeighborsClassifier(n_neighbors = k, weights = w)
            kNN.fit(X_train, y_train)
            
            train_acc = weighted_acc(kNN.predict(X_train), y_train)
            val_acc = weighted_acc(kNN.predict(X_val), y_val)
            
            array.append(kNNclass(kNN, k, w, train_acc=train_acc, val_acc=val_acc))
            
        kNN_models.append(array)
    return kNN_models



def NeuralNetwork(hidden_layer_sizes, seed):
    """
    Creates and stores Neural Network-models with various hyperparamers
    
    Parameters
    ----------
    hidden_layer_sizes  : list of tuples, specifies the amount of hidden layers and their amount of nodes
    seed                : integer, seed for ensuring repeatability in respect to random effects
    
    Returns
    -------
    MLP_models : nested list of class objects, containing model with model data
    """
    MLP_models = []
    
    for i in hidden_layer_sizes:
        MLP = MLPClassifier(i, random_state = seed)
        MLP.fit(X_train,y_train)
        train_acc = weighted_acc(MLP.predict(X_train), y_train)
        val_acc = weighted_acc(MLP.predict(X_val), y_val)
        MLP_models.append(MLPclass(MLP, len(i), i[0], train_acc, val_acc))
    return MLP_models


def DecisionTree(impurity, min_samples_split, seed):
    """
    Creates and stores Decision tree models with various hyperparamers
    
    Parameters
    ----------
    impurity            : list of strings, specifies the type of impurity to be used
    min_samples_split   : list of float, minimum amount of samples required to make a split, given as a fraction of the total amount of data points
    seed                : integer, seed for ensuring repeatability in respect to random effects
    
    Returns
    -------
    DTC_models : nested list of class objects, containing model with model data
    """
    DTC_models = []
    
    for i in impurity:
        array = []
        for m in min_samples_split:
            DTC = DecisionTreeClassifier(criterion = i, min_samples_split = m, random_state = seed)
            DTC.fit(X_train, y_train)
            
            train_acc = weighted_acc(DTC.predict(X_train), y_train)
            val_acc = weighted_acc(DTC.predict(X_val), y_val)
            
            array.append(DTCclass(DTC, i, m, train_acc, val_acc))
            
        DTC_models.append(array)    
    
    return DTC_models
            
            

"""_-_-_-_-_-_-_Main code_-_-_-_-_-_-_"""

""" Fixed parameters for reproducability """
seed = 264
test_size = 0.3
val_size = 0.5
random.seed(seed) #Initialing random seed

""" Loading the data """
X = np.load("MNIST-images.npy")
X = X[0:1000]
X = X.reshape(X.shape[0], 576) #Reshaping the features matrix into a 2D matrix
y = np.load("MNIST-labels.npy")
y = y[0:1000]

#Balancing the data
X, y = balanceData(X, y, seed)

#Splitting the data
X_train, X_val, X_test, y_train, y_val, y_test = dataSets(seed, X, y, test_size, val_size)

#Initialising variable for best model
BestModel = None
BestAccuracy = 0

""" Neural network """
#Create a list of tuples for hidden_layer_sizes
hidden_layers = 6
total_nodes = 2**8
hidden_layer_sizes = []
for i in range(hidden_layers):
    hidden_layer_sizes.append(tuple([int(total_nodes/(2**i))]*2**i))
  
MLP_models = NeuralNetwork(hidden_layer_sizes, seed)

#Create a list of accuracies
train_acc = [model.train_acc for model in MLP_models]
val_acc = [model.val_acc for model in MLP_models]

#If we find a model with better accuracy than the previos model, update variables
if max(val_acc) > BestAccuracy:
    BestModel = MLP_models[val_acc.index(max(val_acc))]
    BestAccuracy = BestModel.val_acc

#Plot the accuracy development of the models
plt.plot(range(1, hidden_layers + 1), train_acc)
plt.plot(range(1, hidden_layers + 1), val_acc)
plt.legend(("Training", "Validation"))
plt.xlabel("Hidden layers")
plt.ylabel("Accuracy")
plt.title("Neural network")

""" k-Nearest Neighbors """
kk = [1, 3, 5, 7, 9, 11, 13]
weight = ["uniform", "distance"]
kNN_models = kNearestNeighbors(kk, weight)

#Plot the accuracy development of the models
fig, axs = plt.subplots(2, 1)
for i in range(2):
    train_acc = [model.train_acc for model in kNN_models[i]]
    val_acc = [model.val_acc for model in kNN_models[i]]
    axs[i].plot(kk, train_acc)
    axs[i].plot(kk, val_acc)
    axs[i].legend(("Training", "Validation"))
    axs[i].set_title(kNN_models[i][0].weight)
    axs[i].set_xlabel("k")
    axs[i].set_ylabel("Accuracy")
    fig.suptitle('k-Nearest Neighbor', fontsize=16)
    plt.tight_layout()
    
    #If we find a model with better accuracy than the previos model, update variables
    if max(val_acc) > BestAccuracy:
        BestModel = kNN_models[i][val_acc.index(max(val_acc))]
        BestAccuracy = BestModel.val_acc
    

""" Decision tree """
impurity = ["gini", "entropy", "log_loss"]
min_samples_split = [2, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002]
DTC_models = DecisionTree(impurity, min_samples_split, seed)

#Plot the accuracy development of the models
fig, axs = plt.subplots(3, 1)
for i in range(3):
    train_acc = [model.train_acc for model in DTC_models[i]]
    val_acc = [model.val_acc for model in DTC_models[i]]
    axs[i].plot(min_samples_split, train_acc)
    axs[i].plot(min_samples_split, val_acc)
    axs[i].set_title(DTC_models[i][0].impurity)
    axs[i].set_xlabel("min_samples_split")
    axs[i].set_ylabel("Accuracy")
    fig.suptitle('Decision tree', fontsize=16)
    plt.tight_layout()
    
    #If we find a model with better accuracy than the previos model, update variables
    if max(val_acc) > BestAccuracy:
        BestModel = DTC_models[i][val_acc.index(max(val_acc))]
        BestAccuracy = BestModel.val_acc


""" Choosing model """
print("The chosen model with the highest validation accuracy is the following:")
print(BestModel)
print("The model's accuracy on test data is " + str(weighted_acc(BestModel.model.predict(X_test), y_test)))
plot_confusion_matrix(BestModel.model, X_test, y_test)
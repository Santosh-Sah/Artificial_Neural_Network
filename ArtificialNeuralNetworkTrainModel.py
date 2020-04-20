# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:04:03 2020

@author: Santosh Sah
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from ArtificialNeuralNetworkUtils import (saveArtificialNeuralNetworkModel, readArtificialNeuralNetworkXTrain, readArtificialNeuralNetworkYTrain,
                                     saveArtificialNeuralNetworkStandardScaler, saveDropoutArtificialNeuralNetworkModel)

"""
Train ArtificialNeuralNetwork model 
"""
def trainArtificialNeuralNetworkModel():
    
    artificialNeuralNetworkStandardScalar = StandardScaler()
    
    X_train = readArtificialNeuralNetworkXTrain()
    y_train = readArtificialNeuralNetworkYTrain()
    
    artificialNeuralNetworkStandardScalar.fit(X_train)
    saveArtificialNeuralNetworkStandardScaler(artificialNeuralNetworkStandardScalar)
    
    X_train = artificialNeuralNetworkStandardScalar.transform(X_train)
    
    #initialising ANN
    artificialNeuralNetwork = Sequential()
    
    #adding the input layer and first hidden layer
    artificialNeuralNetwork.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11))
    
    #adding the second hidden layer
    artificialNeuralNetwork.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
    
    #adding the output layer
    artificialNeuralNetwork.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
    
    #compiling the ANN
    artificialNeuralNetwork.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    #fitting the ANN on the training set
    artificialNeuralNetwork.fit(X_train, y_train, batch_size=10, epochs=100)
    
    #saving themodel as a pickle file
    saveArtificialNeuralNetworkModel(artificialNeuralNetwork)

def crossValidationArtificialNeuralNetworkModel():
    
    #initialising ANN
    artificialNeuralNetwork = Sequential()
    
    #adding the input layer and first hidden layer
    artificialNeuralNetwork.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11))
    
    #adding the second hidden layer
    artificialNeuralNetwork.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
    
    #adding the output layer
    artificialNeuralNetwork.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
    
    #compiling the ANN
    artificialNeuralNetwork.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    return artificialNeuralNetwork

def gridSearchArtificialNeuralNetworkModel(optimizer):
    
    #initialising ANN
    artificialNeuralNetwork = Sequential()
    
    #adding the input layer and first hidden layer
    artificialNeuralNetwork.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11))
    
    #adding the second hidden layer
    artificialNeuralNetwork.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
    
    #adding the output layer
    artificialNeuralNetwork.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
    
    #compiling the ANN
    artificialNeuralNetwork.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    
    return artificialNeuralNetwork

def crossValidation():
    
    X_train = readArtificialNeuralNetworkXTrain()
    y_train = readArtificialNeuralNetworkYTrain()
    
    classifier = KerasClassifier(build_fn=crossValidationArtificialNeuralNetworkModel, batch_size=10, nb_epoch=100)
    accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
    
    print(accuracies)
    """
    [0.78625    0.79000002 0.80000001 0.78250003 0.81625003 0.81
     0.78750002 0.79374999 0.79874998 0.79500002]
    """
    
    mean = accuracies.mean()
    variance = accuracies.std()
    
    print("mean of the acuuracies is: ", str(mean)) #0.79
    print("variance of the acuuracies is: ", str(variance))#0.010

def gridSearchArtificialNeuralNetwork():
    
    classifier = KerasClassifier(build_fn=gridSearchArtificialNeuralNetworkModel)
    parameters = {"batch_size":[25, 32],
                  "nb_epoch":[100,500],
                  "optimizer":['adam','rmsprop']}
    
    gridSearch = GridSearchCV(estimator=classifier, param_grid=parameters, scoring="accuracy", cv = 10)
     
    X_train = readArtificialNeuralNetworkXTrain()
    y_train = readArtificialNeuralNetworkYTrain()
    
    gridSearch = gridSearch.fit(X_train, y_train)
    
    best_parameters = gridSearch.best_params_
    best_score = gridSearch.best_score_
    
    print("best parameters are: ", best_parameters) 
    """
     {'batch_size': 25, 'nb_epoch': 100, 'optimizer': 'adam'}
    """
    print("best score is: " , str(best_score)) #0.796

"""
Train with dropout ArtificialNeuralNetwork model 
"""
def trainWithDropoutArtificialNeuralNetworkModel():
    
    artificialNeuralNetworkStandardScalar = StandardScaler()
    
    X_train = readArtificialNeuralNetworkXTrain()
    y_train = readArtificialNeuralNetworkYTrain()
    
    artificialNeuralNetworkStandardScalar.fit(X_train)
    saveArtificialNeuralNetworkStandardScaler(artificialNeuralNetworkStandardScalar)
    
    X_train = artificialNeuralNetworkStandardScalar.transform(X_train)
    
    #initialising ANN
    artificialNeuralNetwork = Sequential()
    
    #adding the input layer and first hidden layer
    artificialNeuralNetwork.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11))
    #add dropout at input layer
    artificialNeuralNetwork.add(Dropout(p=0.10))
    
    #adding the second hidden layer
    artificialNeuralNetwork.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
    #add dropout at hidden layer
    artificialNeuralNetwork.add(Dropout(p=0.10))
    
    #adding the output layer
    artificialNeuralNetwork.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
    
    #compiling the ANN
    artificialNeuralNetwork.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
    #fitting the ANN on the training set
    artificialNeuralNetwork.fit(X_train, y_train, batch_size=10, epochs=100)
    
    #saving themodel as a pickle file
    saveDropoutArtificialNeuralNetworkModel(artificialNeuralNetwork)


if __name__ == "__main__":
    #trainArtificialNeuralNetworkModel()
    #crossValidation()
    #trainWithDropoutArtificialNeuralNetworkModel()
    gridSearchArtificialNeuralNetwork()
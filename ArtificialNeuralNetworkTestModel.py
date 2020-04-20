# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:04:47 2020

@author: Santosh Sah
"""

from ArtificialNeuralNetworkUtils import (readArtificialNeuralNetworkXTest, readArtificialNeuralNetworkModel, readDropoutArtificialNeuralNetworkModel, 
                                     saveArtificialNeuralNetworkYPred, readArtificialNeuralNetworkStandardScaler, saveDropoutArtificialNeuralNetworkYPred)

"""
test the model on testing dataset
"""
def testArtificialNeuralNetworkModel():
    
    X_test = readArtificialNeuralNetworkXTest()
    artificialNeuralNetworkStandardScaler = readArtificialNeuralNetworkStandardScaler()
    X_test = artificialNeuralNetworkStandardScaler.transform(X_test)
    
    artificialNeuralNetworkModel = readArtificialNeuralNetworkModel()
    
    y_pred = artificialNeuralNetworkModel.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    saveArtificialNeuralNetworkYPred(y_pred)
    
    print(y_pred)

"""
test the dropout model on testing dataset
"""
def testDropoutArtificialNeuralNetworkModel():
    
    X_test = readArtificialNeuralNetworkXTest()
    artificialNeuralNetworkStandardScaler = readArtificialNeuralNetworkStandardScaler()
    X_test = artificialNeuralNetworkStandardScaler.transform(X_test)
    
    artificialNeuralNetworkModel = readDropoutArtificialNeuralNetworkModel()
    
    y_pred = artificialNeuralNetworkModel.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    saveDropoutArtificialNeuralNetworkYPred(y_pred)
    
    print(y_pred)
    
if __name__ == "__main__":
    #testArtificialNeuralNetworkModel()
    testDropoutArtificialNeuralNetworkModel()
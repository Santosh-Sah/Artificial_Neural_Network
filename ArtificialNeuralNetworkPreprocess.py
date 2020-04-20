# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:06:30 2020

@author: Santosh Sah
"""

from ArtificialNeuralNetworkUtils import (importArtificialNeuralNetworkDataset, saveTrainingAndTestingDataset, artificialNeuralNetworkSplitTrainAndTest,
                                          artificialNeuralNetworkEncodingCategoricalVariable, saveArtificialNeuralNetworkLabelencoder_X_1,
                                          saveArtificialNeuralNetworkLabelencoder_X_2, saveArtificialNeuralNetworkOnehotencoder)

def preprocess():
    
    X, y = importArtificialNeuralNetworkDataset("Artificial_Neural_Network_Churn_Modelling.csv")
    
    X, artificialNeuralNetworkLabelencoder_X_1, artificialNeuralNetworkLabelencoder_X_2, artificialNeuralNetworkOnehotencoder = artificialNeuralNetworkEncodingCategoricalVariable(X)
    
    X_train, X_test, y_train, y_test = artificialNeuralNetworkSplitTrainAndTest(X, y)
    
    saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test)
    
    saveArtificialNeuralNetworkLabelencoder_X_1(artificialNeuralNetworkLabelencoder_X_1)
    
    saveArtificialNeuralNetworkLabelencoder_X_2(artificialNeuralNetworkLabelencoder_X_2)
    
    saveArtificialNeuralNetworkOnehotencoder(artificialNeuralNetworkOnehotencoder)
    

if __name__ == "__main__":
    preprocess()
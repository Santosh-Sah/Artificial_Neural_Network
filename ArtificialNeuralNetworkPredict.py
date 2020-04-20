# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:06:57 2020

@author: Santosh Sah
"""

import pandas as pd
from ArtificialNeuralNetworkUtils import readArtificialNeuralNetworkModel, readArtificialNeuralNetworkStandardScaler

def predict():
    
    artificialNeuralNetwork = readArtificialNeuralNetworkModel()
    artificialNeuralNetworkStandardScaler = readArtificialNeuralNetworkStandardScaler()
    
    inputValue = [[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]
    inputValueDataframe = pd.DataFrame(artificialNeuralNetworkStandardScaler.transform(inputValue))
    
    predictedValue = artificialNeuralNetwork.predict(inputValueDataframe.values)
    predictedValue = (predictedValue > 0.5)
    print(predictedValue)

if __name__ == "__main__":
    predict()
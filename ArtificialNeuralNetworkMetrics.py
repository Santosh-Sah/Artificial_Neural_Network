# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:07:48 2020

@author: Santosh Sah
"""

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from ArtificialNeuralNetworkUtils import (readArtificialNeuralNetworkYTest, readArtificialNeuralNetworkYPred, readDropoutArtificialNeuralNetworkYPred)

"""

calculating ArtificialNeuralNetwork confussion matrix

"""
def testArtificialNeuralNetworkConfussionMatrix():
    
    y_test = readArtificialNeuralNetworkYTest()
    y_pred = readArtificialNeuralNetworkYPred()
    
    artificialNeuralNetworkConfussionMatrix = confusion_matrix(y_test, y_pred)
    print(artificialNeuralNetworkConfussionMatrix)
    
    """
    Below is the confussion matrix
    [[1510   85]
    [ 197  208]]    
    """
"""
calculating accuracy score

"""

def testArtificialNeuralNetworkAccuracy():
    
    y_test = readArtificialNeuralNetworkYTest()
    y_pred = readArtificialNeuralNetworkYPred()
    
    artificialNeuralNetworkConfussionAccuracy = accuracy_score(y_test, y_pred)
    
    print(artificialNeuralNetworkConfussionAccuracy) #.859%

"""
calculating classification report

"""

def testArtificialNeuralNetworkClassificationReport():
    
    y_test = readArtificialNeuralNetworkYTest()
    y_pred = readArtificialNeuralNetworkYPred()
    
    artificialNeuralNetworkConfussionClassificationReport = classification_report(y_test, y_pred)
    
    print(artificialNeuralNetworkConfussionClassificationReport)
    
    """
             precision    recall  f1-score   support

          0       0.88      0.95      0.91      1595
          1       0.71      0.51      0.60       405

avg / total       0.85      0.86      0.85      2000
    """

"""

calculating DropoutArtificialNeuralNetwork confussion matrix

"""
def testDropoutArtificialNeuralNetworkConfussionMatrix():
    
    y_test = readArtificialNeuralNetworkYTest()
    y_pred = readDropoutArtificialNeuralNetworkYPred()
    
    artificialNeuralNetworkConfussionMatrix = confusion_matrix(y_test, y_pred)
    print(artificialNeuralNetworkConfussionMatrix)
    
    """
    Below is the confussion matrix
    [[1551   44]
    [ 276  129]]
    """
"""
calculating accuracy score

"""

def testDropoutArtificialNeuralNetworkAccuracy():
    
    y_test = readArtificialNeuralNetworkYTest()
    y_pred = readDropoutArtificialNeuralNetworkYPred()
    
    artificialNeuralNetworkConfussionAccuracy = accuracy_score(y_test, y_pred)
    
    print(artificialNeuralNetworkConfussionAccuracy) #.849%

"""
calculating classification report

"""

def testDropoutArtificialNeuralNetworkClassificationReport():
    
    y_test = readArtificialNeuralNetworkYTest()
    y_pred = readDropoutArtificialNeuralNetworkYPred()
    
    artificialNeuralNetworkConfussionClassificationReport = classification_report(y_test, y_pred)
    
    print(artificialNeuralNetworkConfussionClassificationReport)
    
    """
             precision    recall  f1-score   support

          0       0.85      0.97      0.91      1595
          1       0.75      0.32      0.45       405

avg / total       0.83      0.84      0.81      2000
    """    
if __name__ == "__main__":
    #testArtificialNeuralNetworkConfussionMatrix()
    #testArtificialNeuralNetworkAccuracy()
    #testArtificialNeuralNetworkClassificationReport()
    #testDropoutArtificialNeuralNetworkConfussionMatrix()
    #testDropoutArtificialNeuralNetworkAccuracy()
    testDropoutArtificialNeuralNetworkClassificationReport()
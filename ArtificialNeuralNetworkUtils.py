# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 16:01:32 2020

@author: Santosh Sah
"""
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importArtificialNeuralNetworkDataset(artificialNeuralNetworkDatasetFileName):
    
    artificialNeuralNetworkDataset = pd.read_csv(artificialNeuralNetworkDatasetFileName)
    X = artificialNeuralNetworkDataset.iloc[:, 3:13].values
    y = artificialNeuralNetworkDataset.iloc[:, 13].values
    
    return X, y

def artificialNeuralNetworkSplitTrainAndTest(X,y):
    
    #spliting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return X_train, X_test, y_train, y_test

def artificialNeuralNetworkEncodingCategoricalVariable(X):
    
    artificialNeuralNetworkLabelencoder_X_1 = LabelEncoder()
    artificialNeuralNetworkLabelencoder_X_1.fit(X[:, 1])
    X[:, 1] = artificialNeuralNetworkLabelencoder_X_1.transform(X[:, 1])
    
    artificialNeuralNetworkLabelencoder_X_2 = LabelEncoder()
    artificialNeuralNetworkLabelencoder_X_2.fit(X[:, 2])
    X[:, 2] = artificialNeuralNetworkLabelencoder_X_2.transform(X[:, 2])
    
    artificialNeuralNetworkOnehotencoder = OneHotEncoder(categorical_features = [1])
    artificialNeuralNetworkOnehotencoder.fit(X)
    X = artificialNeuralNetworkOnehotencoder.transform(X).toarray()
    X = X[:, 1:]
    
    return X, artificialNeuralNetworkLabelencoder_X_1, artificialNeuralNetworkLabelencoder_X_2, artificialNeuralNetworkOnehotencoder

"""
Save standard scalar object as a pickel file. This standard scalar object must be used to standardized the dataset for training, testing and new dataset.
To use this standard scalar object we need to read it and then use it.
"""
def saveArtificialNeuralNetworkStandardScaler(artificialNeuralNetworkStandardScalar):
    
    #Write ArtificialNeuralNetworkStandardScaler in a picke file
    with open("ArtificialNeuralNetworkStandardScaler.pkl",'wb') as ArtificialNeuralNetworkStandardScaler_Pickle:
        pickle.dump(artificialNeuralNetworkStandardScalar, ArtificialNeuralNetworkStandardScaler_Pickle, protocol = 2)

"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)
    
    #Write y_train in a picke file
    with open("y_train.pkl",'wb') as y_train_Pickle:
        pickle.dump(y_train, y_train_Pickle, protocol = 2)
    
    #Write y_test in a picke file
    with open("y_test.pkl",'wb') as y_test_Pickle:
        pickle.dump(y_test, y_test_Pickle, protocol = 2)

"""
Save ArtificialNeuralNetworkModel as a pickle file.
"""
def saveArtificialNeuralNetworkModel(artificialNeuralNetworkModel):
    
    #Write ArtificialNeuralNetworkModel as a picke file
    with open("ArtificialNeuralNetworkModel.pkl",'wb') as ArtificialNeuralNetworkModel_Pickle:
        pickle.dump(artificialNeuralNetworkModel, ArtificialNeuralNetworkModel_Pickle, protocol = 2)

"""
read ArtificialNeuralNetworkStandardScalar from pickel file
"""
def readArtificialNeuralNetworkStandardScaler():
    
    #load ArtificialNeuralNetworkStandardScaler object
    with open("ArtificialNeuralNetworkStandardScaler.pkl","rb") as ArtificialNeuralNetworkStandardScaler:
        artificialNeuralNetworkStandardScalar = pickle.load(ArtificialNeuralNetworkStandardScaler)
    
    return artificialNeuralNetworkStandardScalar

"""
read ArtificialNeuralNetworkModel from pickle file
"""
def readArtificialNeuralNetworkModel():
    
    #load ArtificialNeuralNetworkModel model
    with open("ArtificialNeuralNetworkModel.pkl","rb") as ArtificialNeuralNetworkModel:
        artificialNeuralNetworkModel = pickle.load(ArtificialNeuralNetworkModel)
    
    return artificialNeuralNetworkModel

"""
read X_train from pickle file
"""
def readArtificialNeuralNetworkXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readArtificialNeuralNetworkXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
read y_train from pickle file
"""
def readArtificialNeuralNetworkYTrain():
    
    #load y_train
    with open("y_train.pkl","rb") as y_train_pickle:
        y_train = pickle.load(y_train_pickle)
    
    return y_train

"""
read y_test from pickle file
"""
def readArtificialNeuralNetworkYTest():
    
    #load y_test
    with open("y_test.pkl","rb") as y_test_pickle:
        y_test = pickle.load(y_test_pickle)
    
    return y_test

"""
save y_pred as a pickle file
"""

def saveArtificialNeuralNetworkYPred(y_pred):
    
    #Write y_red in a picke file
    with open("y_pred.pkl",'wb') as y_pred_Pickle:
        pickle.dump(y_pred, y_pred_Pickle, protocol = 2)

"""
read y_predt from pickle file
"""
def readArtificialNeuralNetworkYPred():
    
    #load y_test
    with open("y_pred.pkl","rb") as y_pred_pickle:
        y_pred = pickle.load(y_pred_pickle)
    return  y_pred   
        
def saveArtificialNeuralNetworkLabelencoder_X_1(artificialNeuralNetworkLabelencoder_X_1):
    
    #Write artificialNeuralNetworkLabelencoder_X_1 in a picke file
    with open("artificialNeuralNetworkLabelencoder_X_1.pkl",'wb') as artificialNeuralNetworkLabelencoder_X_1_Pickle:
        pickle.dump(artificialNeuralNetworkLabelencoder_X_1, artificialNeuralNetworkLabelencoder_X_1_Pickle, protocol = 2)
"""
read artificialNeuralNetworkLabelencoder_X_1 from pickle file
"""
def readArtificialNeuralNetworkLabelencoder_X_1():
    
    #load y_test
    with open("artificialNeuralNetworkLabelencoder_X_1.pkl","rb") as artificialNeuralNetworkLabelencoder_X_1_pickle:
        artificialNeuralNetworkLabelencoder_X_1 = pickle.load(artificialNeuralNetworkLabelencoder_X_1_pickle)
    return artificialNeuralNetworkLabelencoder_X_1
        
def saveArtificialNeuralNetworkLabelencoder_X_2(artificialNeuralNetworkLabelencoder_X_2):
    
    #Write artificialNeuralNetworkLabelencoder_X_2 in a picke file
    with open("artificialNeuralNetworkLabelencoder_X_2.pkl",'wb') as artificialNeuralNetworkLabelencoder_X_2_Pickle:
        pickle.dump(artificialNeuralNetworkLabelencoder_X_2, artificialNeuralNetworkLabelencoder_X_2_Pickle, protocol = 2)

"""
read artificialNeuralNetworkLabelencoder_X_2 from pickle file
"""
def readArtificialNeuralNetworkLabelencoder_X_2():
    
    #load artificialNeuralNetworkLabelencoder_X_2
    with open("artificialNeuralNetworkLabelencoder_X_2.pkl","rb") as artificialNeuralNetworkLabelencoder_X_2_pickle:
        artificialNeuralNetworkLabelencoder_X_2 = pickle.load(artificialNeuralNetworkLabelencoder_X_2_pickle)
    return artificialNeuralNetworkLabelencoder_X_2
        
def saveArtificialNeuralNetworkOnehotencoder(artificialNeuralNetworkOnehotencoder):
    
    #Write artificialNeuralNetworkOnehotencoder in a picke file
    with open("artificialNeuralNetworkOnehotencoder.pkl",'wb') as artificialNeuralNetworkOnehotencoder_Pickle:
        pickle.dump(artificialNeuralNetworkOnehotencoder, artificialNeuralNetworkOnehotencoder_Pickle, protocol = 2)

"""
read y_predt from pickle file
"""
def readArtificialNeuralNetworkOnehotencoder():
    
    #load artificialNeuralNetworkOnehotencoder
    with open("artificialNeuralNetworkOnehotencoder.pkl","rb") as artificialNeuralNetworkOnehotencoder_pickle:
        artificialNeuralNetworkOnehotencoder = pickle.load(artificialNeuralNetworkOnehotencoder_pickle)    
    return artificialNeuralNetworkOnehotencoder

"""
Save Dropout ArtificialNeuralNetworkModel as a pickle file.
"""
def saveDropoutArtificialNeuralNetworkModel(dropoutArtificialNeuralNetworkModel):
    
    #Write ArtificialNeuralNetworkModel as a picke file
    with open("DropoutArtificialNeuralNetworkModel.pkl",'wb') as DropoutArtificialNeuralNetworkModel_Pickle:
        pickle.dump(dropoutArtificialNeuralNetworkModel, DropoutArtificialNeuralNetworkModel_Pickle, protocol = 2)

"""
read Dropout ArtificialNeuralNetworkModel from pickle file
"""
def readDropoutArtificialNeuralNetworkModel():
    
    #load DropoutArtificialNeuralNetworkModel model
    with open("DropoutArtificialNeuralNetworkModel.pkl","rb") as DropoutArtificialNeuralNetworkModel:
        dropoutArtificialNeuralNetworkModel = pickle.load(DropoutArtificialNeuralNetworkModel)
    
    return dropoutArtificialNeuralNetworkModel

def saveDropoutArtificialNeuralNetworkYPred(dropoutArtificialNeuralNetworkYPred):
    
    #Write dropoutArtificialNeuralNetworkYPred in a picke file
    with open("dropoutArtificialNeuralNetworkYPred.pkl",'wb') as dropoutArtificialNeuralNetworkYPred_Pickle:
        pickle.dump(dropoutArtificialNeuralNetworkYPred, dropoutArtificialNeuralNetworkYPred_Pickle, protocol = 2)
"""
read dropoutArtificialNeuralNetworkYPred from pickle file
"""
def readDropoutArtificialNeuralNetworkYPred():
    
    #load dropoutArtificialNeuralNetworkYPred
    with open("dropoutArtificialNeuralNetworkYPred.pkl","rb") as dropoutArtificialNeuralNetworkYPred_pickle:
        dropoutArtificialNeuralNetworkYPred = pickle.load(dropoutArtificialNeuralNetworkYPred_pickle)
    return dropoutArtificialNeuralNetworkYPred

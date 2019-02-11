
import numpy as np
import csv as cs
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import operator
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from random import randrange
from sklearn.decomposition import PCA
import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.metrics import f1_score
import warnings
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')




def euclidean_distance(instance1, instance2, length):
    distance=np.subtract(instance1,instance2)
    distance=np.dot(distance,distance.T)
    return math.sqrt(distance)






def get_neighbors(training_set, test_sample, k,Label):
    distances = []
    length = len(test_sample)-1
    #print("into get neighbour")
    for x in range(len(training_set)):
        dist = euclidean_distance(test_sample, training_set[x], length)
        distances.append (((int(Label[x]), dist)))
    #print(distances)
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    #print("out of get neighbour")
    return neighbors


def get_majority(neighbors):
    majority = {}
    for x in range(len(neighbors)):
        if (neighbors[x] not in majority):
            majority[neighbors[x]] = 1
        else:
            majority[neighbors[x]] += 1
    return (max(majority.items(), key=operator.itemgetter(1))[0] )



def knn(B,L,A,test_label):

    k = 5
    predictions=[]
    #print(k)
    for x in range(len(A)):
        neighbors = get_neighbors(B, A[x], k,L)
        #print(neighbors)
        result = get_majority(neighbors)
        predictions.append(result)
    count=0
    #print("end of for")
    for i in range (len(A)):
        if(predictions[i]!=test_label[i]):
            count+=1
    accuracy=   ((len(A)-count)/len(A))*100
    #print("my_knn")
    #print (accuracy)
    return accuracy,predictions

    #print (predictions)

def fun_knn(training_set,test_set,training_label,test_label):
    neigh = KNeighborsClassifier(n_neighbors=3)
    pred=neigh.fit(training_set, np.ravel(training_label)).predict(test_set)
    count=0
    for i in range (len(test_set)):
        if(pred[i]!=test_label[i]):
            count+=1

    #print(test_set)
    #print("------------------------------------------------")
    #print(training_set)
    accuracy=   ((len(test_set)-count)/len(test_set))*100
    #print("lib_knn")
    #print (accuracy)
    return accuracy,pred





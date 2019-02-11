
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


curr=os.getcwd()
final=os.path.join(curr,r'output')
if not os.path.exists(final):
    eos.makedirs(final)


def projection(B,d,i):
    curr=os.getcwd()
    final=os.path.join(curr,r'output')
    if not os.path.exists(final):
        eos.makedirs(final)
    B=np.array(B)
    n=len(B)
    m=len(B[0])
    #print(n)
    #print(m)
    C=[]
    D=[]
    rn=np.random.standard_normal(m*d)
    C = np.asmatrix(rn.reshape(m,d))
    D = np.matmul(B,C)
    D= (1/math.sqrt(d))*D
    f=open(final + '/task1_dataset_' + str(i) + '_'+ str(d)+'.txt','w')
    #print(D)
    f.write(str(np.array(D)))
    return np.array(D)

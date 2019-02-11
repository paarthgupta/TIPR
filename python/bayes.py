
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



def find_mean(dict_info) :
    mean_dict={}
    number_label=len(dict_info)
    for key in dict_info:
        mean_dict[key]=[]
        A=dict_info[key]
        A=np.array(A)
        B=[]
        B=A.sum(axis=0)
        num=len(A)
        B=B/num
        mean_dict[key]=list(B)

    return mean_dict



def find_std(dict_info,mean_dict):
    std_dict={}
    number_label=len(dict_info)
    for key in dict_info:
        std_dict[key]=[]
        #mean=mean_dict[key]
        A=dict_info[key]
        #A=np.array(A)
        #minus=np.zeros((1,len(A[0])))
        #minus=[0 for i in range(len(A[0]))]
        #temp=np.zeros((1,len(A[0])))
        #sigma=np.zeros((len(A[0]),len(A[0])))


       # minus=np.array(minus)
        #minus=minus.tolist()
       # print(key)
       # for row in A:
            #D=np.matmul((np.array(row - mean_dict[key])).reshape(len(row),1),np.array(row - mean_dict[key]).reshape(1,len(row)))
            #print(D)
            #sigma=D+sigma
            #print(sigma)
            #print("---------------------------------------------------------------------")
        #    temp=row-mean
        #    minus=minus+pow(temp,2)
            #print(minus)
        #sigma=sigma/len(A)
        #print(sigma)

        #minus=(minus/len(A))

        #minus = [math.sqrt(x) for x in minus[0]]
        #print ((minus))
        #minus=math.sqrt(minus)
        #print(minus)
        #minus=map(math.sqrt,minus/len(A))
        #std_dict[key]=list(minus)
        #minus1=[]
        #print("-------")
        #print(minus[0])
        #print(key)
        #print(":")
        i1=[]
        for j1 in range(len(A[0])):
            i=[]
            for j2 in range(len(A)) :
                i.append(A[j2][j1])
            i1.append(np.std(i))
            std_dict[key]=list(i1)
        #print(i1)




    #B=np.array(std_dict[0])
    #print(B.sum(axis=0))
    #print(B)
    #print(B[25][30])
    #f.write("Me!")

    #print(np.linalg.inv((std_dict[0])))
    #print(np.linalg.det(std_dict[0]))
    #print(std_dict[0])
    return std_dict


def find_priors(dict_info):
    sum=0
    prior={}
    for key in dict_info:
        sum=sum+ len(dict_info[key])
    for key in dict_info:
        prior[key]=len(dict_info[key])/sum
    #print (prior)
    return prior




def bayes(mean_dict,std_dict,test_set,test_label,prior):
    predictions=[]
    #print(len(test_set))
    for x in range(len(test_set)):
        maxi=class_prob(mean_dict,std_dict,test_set[x],prior)
        #print(neighbors)
        #result = get_majority(neighbors)
        predictions.append(maxi)
    count=0
    #sup=0
    #print(len(predictions))
    #print(len(test_set))
    for i in range (len(test_set)):
        #print(predictions[i])
        #print(test_label[i])

        if(predictions[i]!=test_label[i]):
            count+=1
    accuracy=   ((len(test_set)-count)/len(test_set))*100
    #print("my_bayes")
    #print (accuracy)
    return accuracy,predictions
    #print (sup)



def class_prob(mean_dict,std_dict,test_sample,prior):
    prob_dict = {}
    #print(std_dict)
    for key in mean_dict:
        prob_dict[key] = 1
        for i in range(len(mean_dict[key])):
        #for i in range(300):

            mean=mean_dict[key]
            mean=mean[i]
            std= std_dict[key]
            std=std[i]
            x = test_sample[i]
            #print(std,key)

            if (std!=0)  :
                try:
                    if (prob_dict[key]< 1e300) :
                        _ = prob_dict[key]
                        prob_dict[key] = prob_dict[key]  * calcu_prob(mean, std,x)
                except:
                    prob_dict[key] = _

        prob_dict[key]*=prior[key]
        #print(prior[key])
    #print (prob_dict)        #print(prob_dict[key])
    #print(prob_dict)
    return (max(prob_dict.items(), key=operator.itemgetter(1))[0] )


def calcu_prob(mean, std , test_sample):


    exponent = math.exp(-(math.pow(test_sample - mean,2)/(2*math.pow(std,2))))
    return ((1 / (math.sqrt(2*math.pi) * std)) * exponent)




def fun_bayes(training_set,test_set,training_label,test_label):

    gnb = GaussianNB()
    pred = gnb.fit(training_set, np.ravel(training_label)).predict(test_set)

    count=0
    for i in range (len(test_set)):
        if(pred[i]!=test_label[i]):
            count+=1
    accuracy=   ((len(test_set)-count)/len(test_set))*100

    #print("lib_bayes")
    #print (accuracy)
    return accuracy,pred

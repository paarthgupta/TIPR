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
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.metrics import f1_score
import warnings

from LocalitySensitiveHashing import *


curr=os.getcwd()
final=os.path.join(curr,r'output')
if not os.path.exists(final):
    eos.makedirs(final)


#file=open(final + '/accuracy.txt','w')
file1=open(final + '/accuracy_oiu.txt','w')




def process_twitter(X2,Y2) :


    # X2.shape, Y2.shape, X2.info()
    #print(X2)

    Y2=np.array(Y2)
    #print(Y2[5998][0])

    # Converting into bag-of-word
    all_words = set()
    local_ls = []
    dict_plus={}
    dict_minus={}
    dict_zero={}
    dictwhole={}

    for indx,stmt in X2.iterrows():

        for word in stmt[0].strip().split():
            if word not in dictwhole:
                dictwhole[word]=1
            dictwhole[word]+=1
    dictwhole={key:value for key,value in dictwhole.items() if value>1}
    #print(len(dictwhole))
    for indx,stmt in X2.iterrows():

        for word in stmt[0].strip().split():
            if (word not in stopwords.words('english') and word in dictwhole) :
                if Y2[indx][0]==1:
                    if word in dict_plus:
                        dict_plus[word]+=1
                    else:
                        dict_plus[word]=1

                elif Y2[indx][0]==0:
                    if word in dict_zero:
                        dict_zero[word]+=1
                    else:
                        dict_zero[word]=1

                if Y2[indx][0]==-1:
                    if word in dict_minus:
                        dict_minus[word]+=1
                    else:
                        dict_minus[word]=1
    #print(len(dict_plus))
    return(dict_minus,dict_zero,dict_plus)




def split_data(dataset,label):
    size_t = int(len(dataset) *0.80 )
    #print(size_t)
    training = []
    training_label = []

    dataset1 = np.array(dataset)
    dataset1=list(dataset1)
    #print(dataset1)#.pop(0))
    #print("------------------")
    dataset2=list(label)
    #print(dataset1)
    while len(training) < size_t:
        #index = random.randrange(len(dataset1))
        index=0
        #print(np.array(dataset1))
        #print("------------------")
        #print((dataset2.pop(index)))
        training_label.append(list(dataset2.pop(index)))
        training.append(list(dataset1.pop(index)))
    #print((training))
    return(training, dataset1,training_label,dataset2)

def start():
    X2 = pd.read_csv("E:\\twitter.txt",header=None)
    Y2 = pd.read_csv("E:\\twitter_label.txt",header=None)
    Y2=np.array(Y2)
    #training_set, test_set,training_label,test_label = split_data(np.array((X2)),np.array((Y2)))
    list_k_fold= module1.man_split(np.array(X2),Y2,5)
    accuracy=0
    micro_bayes=0
    macro_bayes=0
    for k1 in range(5):
        #print(k)
        #print("k1=",end=' ')
        #print(k1)



        test_set=[]
        training_set=[]
        training_label=[]
        test_label=[]
        prior={}





        label1=[]
    		#print(list_k_fold)
        for i in range(5):
            if i==k1:
                label1.extend(list_k_fold[i])
        for i2 in range(len(Y2)):
            if i2 in label1:
                test_set.append(np.array(X2)[i2])
                test_label.append(Y2[i2])
            else:
                training_set.append(np.array(X2)[i2])
                #print(trainset)
                training_label.append(Y2[i2])

        dict_info={}
        dict_info = module1.form_dict(training_set,training_label)

        test_set=np.array(test_set)
        test_label=np.array(test_label)
        #print("---------------------------------" )
        training_set=np.array(training_set)
        training_label=np.array(training_label)

        training_set=pd.DataFrame(data=training_set)
        training_label=pd.DataFrame(data=training_label)

        size=len(test_set)
        #print(len(dataset2))
        priors=module1.find_priors(dict_info)


        dict_minus={}
        dict_zero={}
        dict_plus={}
        dict_minus,dict_zero,dict_plus = process_twitter(training_set,training_label)
        #print(len(dict_plus))

        predictions=[]
        temp=1
        maxi=0
        prob=1

        #print(dict_plus[1])
        #dict_plus = sorted(dict_plus.items(), key=operator.itemgetter(1))
        #dict_plus = sorted(dict_zero.items(), key=operator.itemgetter(1))
        #dict_plus = sorted(dict_minus.items(), key=operator.itemgetter(1))

        #dict_plus={key:value for key,value in dict_plus.items() if value>=1}
        #dict_minus={key:value for key,value in dict_minus.items() if value>=1}
        #dict_zero={key:value for key,value in dict_zero.items() if value>=1}


        size_plus= sum(dict_plus.values())
        size_minus= sum(dict_minus.values())
        size_zero= sum(dict_zero.values())



        for i in range(size):

            maxi=0

            for k in range (-1,2):

                prob=priors[k]

                for word in (str(test_set[i][0])).split():


                    if(k==-1):
                        if(word in dict_minus) :
                            prob=prob*((dict_minus[word]+1)/(size_minus+2998))
                        else:
                            prob=prob*((1)/(size_minus+2998))


                    if(k==0):
                        if(word in dict_zero) :
                            prob=prob*((dict_zero[word]+1)/(size_zero+2998))
                        else:
                            prob=prob*((1)/(size_zero+2998))
                    if(k==1):
                        if(word in dict_plus) :
                            prob=prob*((dict_plus[word]+1)/(size_plus+2998))
                        else:
                            prob=prob*((1)/(size_plus+2998))

                if(prob>maxi):
                    maxi=prob
                    temp=k
            predictions.append(temp)
        count=0

        for i in range(size):
            if(predictions[i]!=test_label[i]):
                    count+=1
        accuracy1=   ((size-count)/size)*100
        accuracy+=accuracy1
        #print(accuracy)
        micro_bayes+=f1_score(test_label,predictions,average='micro')
        macro_bayes+=f1_score(test_label,predictions,average='macro')

    file1.write("Test Accuracy on twitter using second bayes  ::" + str(accuracy/5) +"\n")


    file1.write("Test Macro F1 Score on twitter using second bayes ::" + str(macro_bayes/5) +"\n")


    file1.write("Test Micro F1 Score on twitter using second bayes ::" + str(micro_bayes/5) +"\n")

    file1.close()


start();

















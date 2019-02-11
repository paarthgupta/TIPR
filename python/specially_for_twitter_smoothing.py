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

fileq=open(final + '/task3_smooth_bayes.txt','w')



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

def form_dict(training_set,training_label):
    dict_info = {}
    for i in range(len(training_set)):
        vector = training_set[i]
        if (training_label[i][0] not in dict_info):
            dict_info[training_label[i][0]] = []
        dict_info[training_label[i][0]].append(vector)
    return dict_info

def man_split(training,A,k):
    k=5
    dict={}
    i=0
    for Z in A:
        if Z[0] in dict:
            dict[Z[0]].add(i)
        else:
            dict[Z[0]] = {i}
        i=i+1
    classes={}
    for Z in dict:
        l=len(list(dict[Z]))
        copy=list(dict[Z])
        split = list()
        size = (int(l / k))
        for i in range(k):
            fold = list()
            while len(fold) < size:
                i = randrange(len(copy))
                fold.append(copy.pop(i))
            split.append(fold)
        classes[Z]=split
    listfold=[0 for i in range(k)]
    for i in range(k):
        small=[]
        for Z in classes:
            small.extend(classes[Z][i])
        listfold[i]=list(small)
    return (listfold)



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
    list_k_fold= man_split(np.array(X2),Y2,5)
    accuracy=0
    micro=0
    macro=0
    training_set=np.array(X2)
    training_label=np.array(Y2)


    dict_info={}
    dict_info = form_dict(training_set,training_label)

    test_set=np.array(test_set)
    test_label=np.array(test_label)
    #print("---------------------------------" )
    training_set=np.array(training_set)
    training_label=np.array(training_label)

    training_set=pd.DataFrame(data=training_set)
    training_label=pd.DataFrame(data=training_label)

    size=len(test_set)
    #print(len(dataset2))
    priors=find_priors(dict_info)


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
    micro1=f1_score(test_label,predictions,average='micro')
    micro+=micro1
    macro1=f1_score(test_label,predictions,average='macro')
    macro+=macro1
#print(accuracy/5)
#print(micro/5)
#print(macro/5)
    fileq.write("Test Accuracy on twitter using smoothing bayes::" + str(accuracy) +"\n")
    fileq.write("Test Macro F1 Score on twitter using smoothing bayes::" + str(macro) +"\n")
    fileq.write("Test Micro F1 Score on twitter using smoothing bayes::" + str(micro) +"\n")
    fileq.close()
#print("finally")




#start()














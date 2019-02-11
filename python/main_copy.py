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
#import nn
#import abc
import datetime
import projections

def accuracy(predictions,test_label):
    if(predictions[i]!=test_label[i]):
        count+=1
    accuracy=   ((len(test_set)-count)/len(test_set))*100
    #print("my_bayes")
    #print (accuracy)
    return accuracy,predictions
    #print (sup)

    return accuracy and predictions






curr=os.getcwd()
final=os.path.join(curr,r'output')
if not os.path.exists(final):
    eos.makedirs(final)


file=open(final + '/accuracy.txt','w')
file1=open(final + '/accuracy_oiu.txt','w')





def plot(x,y,k):
    #print(y)
    if k<4:
        a="dolphins"
    else:
        a="pubmed"
    for i in range (len(y)):
        plt.plot(x, y[i], label=a)
        #plt.plot(x, x, label='dolphin')
        #plt.legend()
    plt.savefig('file'+str(k)+'.png')
    #plt.show()
    plt.clf

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



def load_data(filename):
    df = pd.read_csv(filename, delimiter=' ',header=None)
    A = np.array(df)
    #print('Loading done')
    return A


#def new_split(data,label_data,K=10):
 #   'Default is 10 fold cross-validation'
  #  classes = set(label_data)
   # class_to_index = {}
    #for index,label in label_data.iterrows():
#        label = label[0]
 #       if label in class_to_index:
  #          class_to_index[label].add(index)
   #     else:
    #        class_to_index[label] = {index}


#    for i in class_to_index:
 #       class_to_index[i] = list(class_to_index[i])
  #      np.random.shuffle(class_to_index[i])


   # class_to_index2 = {}
    #for cls in class_to_index:
     #   a = int(np.round(len(class_to_index[cls])/K))
      #  class_to_index2[cls] = []
       # for fold in range(K):
        #    test_indices  = class_to_index[cls][a*fold:a*(fold+1)]
         #   train_indices = class_to_index[cls][0:a*fold] + class_to_index[cls][a*(fold+1):]
          #  class_to_index2[cls].append((train_indices,test_indices))
#        np.random.shuffle(class_to_index2[cls])
 #
  #  index_set = []
   # for i in range(K):
    #    train,test = set(),set()
     #   for cls in class_to_index2:
      #      _ = class_to_index2[cls][i]
       #     train.update(set(_[0]))
        #    test.update (set(_[1]))
#        index_set.append((list(train),list(test)))

 #   return index_set



def form_dict(training_set,training_label):
    dict_info = {}
    for i in range(len(training_set)):
        vector = training_set[i]
        if (training_label[i][0] not in dict_info):
            dict_info[training_label[i][0]] = []
        dict_info[training_label[i][0]].append(vector)
    return dict_info

#def split_data(dataset,label):
 #   size_t = int(len(dataset) *0.80 )
  #  training = []
   # training_label = []
    #dataset1 = list(dataset)
#    dataset2=list(label)
 #   while len(training) < size_t:
  #      #index = random.randrange(len(dataset1))
   #     index=0
    #    training_label.append(list(dataset2.pop(index)))
     #   training.append(list(dataset1.pop(index)))
#    return[training, dataset1,training_label,dataset2]




def get_neighbors(training_set, test_sample, k,Label):
    distances = []
    length = len(test_sample)-1
    #print("into get neighbour")
    for x in range(len(training_set)):
        dist = euclidean_distance(test_sample, training_set[x], length)
        distances.append (((int(Label[x]), dist)))
        #print(x)
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


def pca (X,components):
    return pd.DataFrame(PCA(n_components=components).fit_transform(X))



def knn(B,L,A,test_label):

    k = 5
    predictions=[]
    #print(k)
    for x in range(len(A)):
        neighbors = get_neighbors(B, A[x], k,L)
        #print(neighbors)
        result = get_majority(neighbors)
        predictions.append(result)
        #print(x)
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

best=0
second_best=0

def find_priors(dict_info):
    global best
    global second_best
    sum=0
    prior={}
    for key in dict_info:
        sum=sum+ len(dict_info[key])
    for key in dict_info:
        prior[key]=len(dict_info[key])/sum
    #print (prior)
    best=max(prior.items(), key=operator.itemgetter(1))[0]


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
    global best
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
                        prob_dict[key] = prob_dict[key]  *0.5* calcu_prob(mean, std,x)
                except:
                    prob_dict[key] = _

        prob_dict[key]*=prior[key]
        #print(prior[key])
    #print (prob_dict)        #print(prob_dict[key])
    #print(prob_dict)

    return (max(prob_dict.items(), key=operator.itemgetter(1))[0])


def calcu_prob(mean, std , test_sample):


    exponent = math.exp(-(math.pow(test_sample - mean,2)/(2*math.pow(std,2))))
    return ((1 / (math.sqrt(2*math.pi) * std)) * exponent)





# knn start ------------------------------------------------------------------------------

def euclidean_distance(instance1, instance2, length):

    distance=np.subtract(instance1,instance2)
    distance=np.dot(distance,distance.T)
    return math.sqrt(distance)













def projection(B,d,i):
    np.array(B)
    n=len(B)
    m=len(B[0])
    #print(n)
    #print(m)
    C=[]
    D=[]
    rn=np.random.standard_normal(m*d)
    C = np.asmatrix(rn.reshape(m,d))
    #print(np.shape(rn))
    #print(np.shape(B))
    #print(np.shape(C))
    D = np.matmul(B,C)
    D= (1/math.sqrt(d))*D
    f=open(final + '/task1_dataset_' + str(i) + '_'+ str(d)+'.txt','w')
    #print(D)
    f.write(str(np.array(D)))
    return np.array(D)






def process_twitter():
    X2 = pd.read_csv("E:\\twitter.txt",header=None)
    Y2 = pd.read_csv("E:\\twitter_label.txt",header=None)
    all_words = set()
    local_ls = []
    for indx,stmt in X2.iterrows():
        local = {}
        for word in stmt[0].strip().split():
            if word not in stopwords.words('english') :
                if word in local:
                    local[word] += 1
                else:
                    local[word]  = 1
        local_ls.append(local)
        all_words.update(local)
    mat = [[(local[word] if word in local else 0) for word in all_words] for local in local_ls]

    X2=np.array(mat)

    return(X2)

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



#------------------------------------twitter end---------------------------------------





# knn end -------------------------------------------------------------------------------

# ----------------------twitter---------------------------------------                    ------------------------------------

def start():

    for i in range(3):

        #i=1
        if (i==0):

            d=2

            train=load_data("E:\\dolphins.csv")

            m,n = train.shape
            labels=load_data("E:\\dolphins_label.csv")
            x=[]
            a=[]
            b=[]
            c=[]
            list1=[]
            list2=[]
            list3=[]
            list4=[]
            list5=[]
            list6=[]
            list7=[]
            list8=[]
            list9=[]
            list10=[]
            list11=[]
            list12=[]
            while (d<=int(n/2)):
                print(d)
                #train_copy=train
                train_copy=projection(train,d,1)


                #training_set,test_set,training_label,test_label
                list_k_fold= man_split(train_copy,labels,5)
                k=5




                acc_bayes=0
                acc_funbayes=0
                acc_knn=0
                acc_funknn=0

                micro_bayes=0
                micro_funbayes=0
                micro_knn=0
                micro_funknn=0
                macro_bayes=0
                macro_funbayes=0
                macro_knn=0
                macro_funknn=0
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
                    for i in range(k):
                        if i==k1:
                            label1.extend(list_k_fold[i])
                    for i2 in range(len(labels)):
                        if i2 in label1:
                            test_set.append(train_copy[i2])
                            test_label.append(labels[i2])
                        else:
                            training_set.append(train_copy[i2])
                            #print(trainset)
                            training_label.append(labels[i2])

                    dict_info={}
                    dict_info = form_dict(training_set,training_label)
                    mean_dict=find_mean(dict_info)
                    std_dict = find_std (dict_info,mean_dict)
                    prior=find_priors(dict_info)

                    #print(len(training_set))
                    #print(len(test_set))




                    _,predictions0 = fun_bayes(training_set,test_set,training_label,test_label)

                    acc_funbayes+= _
                    micro_funbayes+=f1_score(test_label,predictions0,average='micro')
                    macro_funbayes+=f1_score(test_label,predictions0,average='macro')


                    _,predictions1=fun_knn(training_set,test_set,training_label,test_label)
                    acc_funknn+= _
                    micro_funknn+=f1_score(test_label,predictions1,average='micro')
                    macro_funknn+=f1_score(test_label,predictions1,average='macro')


                    _,predictions2=knn(training_set,training_label,test_set,test_label)
                    acc_knn+= _


                    micro_knn+=f1_score(test_label,predictions2,average='micro')
                    macro_knn+=f1_score(test_label,predictions2,average='macro')


                    _,predictions3=bayes(mean_dict,std_dict,test_set,test_label,prior)
                    acc_bayes+= _
                    micro_bayes+=f1_score(test_label,predictions3,average='micro')
                    macro_bayes+=f1_score(test_label,predictions3,average='macro')

                file.write("Test Accuracy on dolphin using inbuilt bayes and d =" + str(d) + "::" + str(acc_funbayes/5) +"\n")
                file.write("Test Accuracy on dolphin using inbuilt knn and d =" + str(d) + "::" + str(acc_funknn/5) +"\n")
                file.write("Test Accuracy on dolphin using my bayes and d =" + str(d) + "::" + str(acc_bayes/5) +"\n")
                file.write("Test Accuracy on dolphin using my knn and d =" + str(d) + "::" + str(acc_knn/5) +"\n \n")

                file.write("Test Macro F1 Score on dolphin using inbuilt bayes and d =" + str(d) + "::" + str(macro_funbayes/5) +"\n")
                file.write("Test Macro F1 Score on dolphin using inbuilt knn and d =" + str(d) + "::" + str(macro_funknn/5) +"\n")
                file.write("Test Macro F1 Score on dolphin using my bayes and d =" + str(d) + "::" + str(macro_bayes/5) +"\n")
                file.write("Test Macro F1 Score on dolphin using my knn and d =" + str(d) + "::" + str(macro_knn/5) +"\n \n")

                file.write("Test Micro F1 Score on dolphin using inbuilt bayes and d =" + str(d) + "::" + str(micro_funbayes/5) +"\n")
                file.write("Test Micro F1 Score on dolphin using inbuilt knn and d =" + str(d) + "::" + str(micro_funknn/5) +"\n")
                file.write("Test Micro F1 Score on dolphin using my bayes and d =" + str(d) + "::" + str(micro_bayes/5) +"\n")
                file.write("Test Micro F1 Score on dolphin using my knn and d =" + str(d) + "::" + str(micro_knn/5) +"\n \n")




#                print(acc_funbayes/5)
 #               print(acc_funknn/5)
  #              print(acc_knn/5)
   #             print(micro_bayes/5)
    #            print(micro_funbayes/5)
     #           print(micro_funknn/5)
      #          print(micro_knn/5)
       #         print(macro_bayes/5)
        #        print(macro_funbayes/5)
         #       print(macro_funknn/5)
          #      print(macro_knn/5)
                x.append(d)
                list1.append(acc_bayes/5)

                list2.append(acc_funbayes/5)
                list3.append(acc_funknn/5)
                list4.append(acc_knn/5)
                list5.append(micro_bayes/5)
                list6.append(micro_funbayes/5)
                list7.append(micro_funknn/5)
                list8.append(micro_knn/5)
                list9.append(macro_bayes/5)
                list10.append(macro_funbayes/5)
                list11.append(macro_funknn/5)
                list12.append(macro_knn/5)


                d=d*2
            a.append(list1)
            a.append(list2)
            a.append(list3)
            a.append(list4)
            b.append(list5)
            b.append(list6)
            b.append(list7)
            b.append(list8)
            c.append(list9)
            c.append(list10)
            c.append(list11)
            c.append(list12)
            plot(x,a,1)
            plot(x,b,2)
            plot(x,c,3)

        elif(i==1):

            d=2
            list1=[]
            lidt2=[]
            list3=[]

            train=load_data("E:\\pubmed.csv")

            m,n = train.shape
            labels=load_data("E:\\pubmed_label.csv")
            x=[]
            a=[]
            b=[]
            c=[]
            list1=[]
            list2=[]
            list3=[]
            list4=[]
            list5=[]
            list6=[]
            list7=[]
            list8=[]
            list9=[]
            list10=[]
            list11=[]
            list12=[]

            while (d<=int(n/2)):
                print(d)
                #train_copy=train
                train_copy=projection(train,d,2)


                #training_set,test_set,training_label,test_label
                list_k_fold= man_split(train_copy,labels,5)
                k=5



                acc_bayes=0
                acc_funbayes=0
                acc_knn=0
                acc_funknn=0


                micro_bayes=0
                micro_funbayes=0
                micro_knn=0
                micro_funknn=0
                macro_bayes=0
                macro_funbayes=0
                macro_knn=0
                macro_funknn=0
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
                    for i in range(k):
                        if i==k1:
                            label1.extend(list_k_fold[i])
                    for i2 in range(len(labels)):
                        if i2 in label1:
                            test_set.append(train_copy[i2])
                            test_label.append(labels[i2])
                        else:
                            training_set.append(train_copy[i2])
                            #print(trainset)
                            training_label.append(labels[i2])

                    dict_info={}
                    dict_info = form_dict(training_set,training_label)
                    mean_dict=find_mean(dict_info)
                    std_dict = find_std (dict_info,mean_dict)
                    prior=find_priors(dict_info)

                    _,predictions0=fun_bayes(training_set,test_set,training_label,test_label)
                    acc_funbayes+= _
                    micro_funbayes+=f1_score(test_label,predictions0,average='micro')
                    macro_funbayes+=f1_score(test_label,predictions0,average='macro')

                    #print("1")

                    _,predictions1=fun_knn(training_set,test_set,training_label,test_label)
                    acc_funknn+= _
                    micro_funknn+=f1_score(test_label,predictions1,average='micro')
                    macro_funknn+=f1_score(test_label,predictions1,average='macro')

                    #print("2")
                    _,predictions2=knn(training_set,training_label,test_set,test_label)
                    acc_knn+= _
                    micro_knn+=f1_score(test_label,predictions2,average='micro')
                    macro_knn+=f1_score(test_label,predictions2,average='macro')

                    #print("3")
                    _,predictions3=bayes(mean_dict,std_dict,test_set,test_label,prior)
                    acc_bayes+= _
                    micro_bayes+=f1_score(test_label,predictions3,average='micro')
                    macro_bayes+=f1_score(test_label,predictions3,average='macro')

                    #print("4")
                file.write("Test Accuracy on pubmed using inbuilt bayes and d =" + str(d) + "::" + str(acc_funbayes/5) +"\n")
                file.write("Test Accuracy on pubmed using inbuilt knn and d =" + str(d) + "::" + str(acc_funknn/5) +"\n")
                file.write("Test Accuracy on pubmed using my bayes and d =" + str(d) + "::" + str(acc_bayes/5) +"\n")
                file.write("Test Accuracy on pubmed using my knn and d =" + str(d) + "::" + str(acc_knn/5) +"\n \n")

                file.write("Test Macro F1 Score on pubmed using inbuilt bayes and d =" + str(d) + "::" + str(macro_funbayes/5) +"\n")
                file.write("Test Macro F1 Score on pubmed using inbuilt knn and d =" + str(d) + "::" + str(macro_funknn/5) +"\n")
                file.write("Test Macro F1 Score on pubmed using my bayes and d =" + str(d) + "::" + str(macro_bayes/5) +"\n")
                file.write("Test Macro F1 Score on pubmed using my knn and d =" + str(d) + "::" + str(macro_knn/5) +"\n \n")

                file.write("Test Micro F1 Score on pubmed using inbuilt bayes and d =" + str(d) + "::" + str(micro_funbayes/5) +"\n")
                file.write("Test Micro F1 Score on pubmed using inbuilt knn and d =" + str(d) + "::" + str(micro_funknn/5) +"\n")
                file.write("Test Micro F1 Score on pubmed using my bayes and d =" + str(d) + "::" + str(micro_bayes/5) +"\n")
                file.write("Test Micro F1 Score on pubmed using my knn and d =" + str(d) + "::" + str(micro_knn/5) +"\n \n")


                print(acc_bayes/5)
                print(acc_funbayes/5)
                print(acc_funknn/5)
                print(acc_knn/5)
                print(micro_bayes/5)
                print(micro_funbayes/5)
                print(micro_funknn/5)
                print(micro_knn/5)
                print(macro_bayes/5)
                print(macro_funbayes/5)
                print(macro_funknn/5)
                print(macro_knn/5)

                x.append(d)
                list1.append(acc_bayes/5)
                #print(acc_bayes/5)
                list2.append(acc_funbayes/5)
                #print(acc_funbayes/5)
                list3.append(acc_funknn/5)
                list4.append(acc_knn/5)
                list5.append(micro_bayes/5)
                list6.append(micro_funbayes/5)
                list7.append(micro_funknn/5)
                list8.append(micro_knn/5)
                list9.append(macro_bayes/5)
                list10.append(macro_funbayes/5)
                list11.append(macro_funknn/5)
                list12.append(macro_knn/5)


                d=d*2
            a.append(list1)
            a.append(list2)
            a.append(list3)
            a.append(list4)
            b.append(list5)
            b.append(list6)
            b.append(list7)
            b.append(list8)
            c.append(list9)
            c.append(list10)
            c.append(list11)
            c.append(list12)
            plot(x,a,4)
            plot(x,b,5)
            plot(x,c,6)



        else :
            d=1024
            list1=[]
            lidt2=[]
            list3=[]
            train=process_twitter()


            m,n = train.shape
            #print(end)
            #print(B)
            #projection(B,i)
            labels=load_data("E:\\twitter_label.txt")
            x=[]
            a=[]
            b=[]
            c=[]
            list1=[]
            list2=[]
            list3=[]
            list4=[]
            list5=[]
            list6=[]
            list7=[]
            list8=[]
            list9=[]
            list10=[]
            list11=[]
            list12=[]

            while (d<=int(n/2)):
                print(d)
                train_copy=train
                #train_copy=projection(train,d,3)



                list_k_fold= man_split(train_copy,labels,5)
                k=5
                acc_bayes=0
                acc_funbayes=0
                acc_knn=0
                acc_funknn=0
                micro_bayes=0
                micro_funbayes=0
                micro_knn=0
                micro_funknn=0
                macro_bayes=0
                macro_funbayes=0
                macro_knn=0
                macro_funknn=0
                for k1 in range(5):
                    #print(k)
                    #print("k1=",end=' ')
                    #print(k1)
                    prior={}
                    dic1={}
                    dic2={}
                    test_set=[]
                    training_set=[]
                    training_label=[]
                    test_label=[]
                    prior={}
                    dic={}


                    label1=[]
            			#print(list_k_fold)
                    for i in range(k):
                        if i==k1:
                            label1.extend(list_k_fold[i])
                    for i2 in range(len(labels)):
                        if i2 in label1:
                            test_set.append(train_copy[i2])
                            test_label.append(labels[i2])
                        else:
                            training_set.append(train_copy[i2])
                            #print(trainset)
                            training_label.append(labels[i2])

                    dict_info={}
                    dict_info = form_dict(training_set,training_label)
                    mean_dict=find_mean(dict_info)
                    std_dict = find_std (dict_info,mean_dict)
                    prior=find_priors(dict_info)

                    #print(len(training_set))
                    #print(len(test_set))

                    _,predictions0=fun_bayes(training_set,test_set,training_label,test_label)
                    acc_funbayes+= _
                    micro_funbayes+=f1_score(test_label,predictions0,average='micro')
                    macro_funbayes+=f1_score(test_label,predictions0,average='macro')

                    #print(micro_bayes/5)

                    _,predictions1=fun_knn(training_set,test_set,training_label,test_label)
                    acc_funknn+= _
                    micro_funknn+=f1_score(test_label,predictions1,average='micro')
                    macro_funknn+=f1_score(test_label,predictions1,average='macro')

                    #print(micro_bayes/5)

                    _,predictions2=knn(training_set,training_label,test_set,test_label)
                    acc_knn+= _
                    micro_knn+=f1_score(test_label,predictions2,average='micro')
                    macro_knn+=f1_score(test_label,predictions2,average='macro')

                    #print(micro_bayes/5)

                    _,predictions3=bayes(mean_dict,std_dict,test_set,test_label,prior)
                    acc_bayes+= _
                    micro_bayes+=f1_score(test_label,predictions3,average='micro')
                    macro_bayes+=f1_score(test_label,predictions3,average='macro')

                    #print(micro_bayes/5)





                file.write("Test Accuracy on twitter using inbuilt bayes and d =" + str(d) + "::" + str(acc_funbayes/5) +"\n")
                file.write("Test Accuracy on twitter using inbuilt knn and d =" + str(d) + "::" + str(acc_funknn/5) +"\n")
                file.write("Test Accuracy on twitter using my bayes and d =" + str(d) + "::" + str(acc_bayes/5) +"\n")
                file.write("Test Accuracy on twitter using my knn and d =" + str(d) + "::" + str(acc_knn/5) +"\n \n")

                file.write("Test Macro F1 Score on twitter using inbuilt bayes and d =" + str(d) + "::" + str(macro_funbayes/5) +"\n")
                file.write("Test Macro F1 Score on twitter using inbuilt knn and d =" + str(d) + "::" + str(macro_funknn/5) +"\n")
                file.write("Test Macro F1 Score on twitter using my bayes and d =" + str(d) + "::" + str(macro_bayes/5) +"\n")
                file.write("Test Macro F1 Score on twitter using my knn and d =" + str(d) + "::" + str(macro_knn/5) +"\n \n")

                file.write("Test Micro F1 Score on twitter using inbuilt bayes and d =" + str(d) + "::" + str(micro_funbayes/5) +"\n")
                file.write("Test Micro F1 Score on twitter using inbuilt knn and d =" + str(d) + "::" + str(micro_funknn/5) +"\n")
                file.write("Test Micro F1 Score on twitter using my bayes and d =" + str(d) + "::" + str(micro_bayes/5) +"\n")
                file.write("Test Micro F1 Score on twitter using my knn and d =" + str(d) + "::" + str(micro_knn/5) +"\n \n")

                print(acc_bayes/5)
                print(acc_funbayes/5)
                print(acc_funknn/5)
                print(acc_knn/5)
                print(micro_bayes/5)
                print(micro_funbayes/5)
                print(micro_funknn/5)
                print(micro_knn/5)
                print(macro_bayes/5)
                print(macro_funbayes/5)
                print(macro_funknn/5)
                print(macro_knn/5)
                x.append(d)
                list1.append(acc_bayes/5)
                #print(acc_bayes/5)
                list2.append(acc_funbayes/5)
                #print(acc_funbayes/5)
                list3.append(acc_funknn/5)
                list4.append(acc_knn/5)
                list5.append(micro_bayes/5)
                list6.append(micro_funbayes/5)
                list7.append(micro_funknn/5)
                list8.append(micro_knn/5)
                list9.append(macro_bayes/5)
                list10.append(macro_funbayes/5)
                list11.append(macro_funknn/5)
                list12.append(macro_knn/5)

                d=d*2
            a.append(list1)
            a.append(list2)
            a.append(list3)
            a.append(list4)
            b.append(list5)
            b.append(list6)
            b.append(list7)
            b.append(list8)
            c.append(list9)
            c.append(list10)
            c.append(list11)
            c.append(list12)
            plot(x,a,7)
            plot(x,b,8)
            plot(x,c,9)

            file.close()

def dolphins():

    #print("done")
    #print(datetime.datetime.now())

    train=load_data("E:\\dolphins.csv")
    m,n = train.shape
    labels=load_data("E:\\dolphins_label.csv")
    train_copy=train
    list_k_fold= man_split(train_copy,labels,5)
    acc_bayes=0
    acc_funbayes=0
    acc_knn=0
    acc_funknn=0

    #print("done")

    micro_bayes=0
    micro_funbayes=0
    micro_knn=0
    micro_funknn=0
    macro_bayes=0
    macro_funbayes=0
    macro_knn=0
    macro_funknn=0
    for k1 in range(5):
        #print(k1)
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

        #print("done")

        for i2 in range(len(labels)):
            if i2 in label1:
                test_set.append(train_copy[i2])
                test_label.append(labels[i2])
            else:
                training_set.append(train_copy[i2])
                #print(trainset)
                training_label.append(labels[i2])

        #print("done")

        dict_info={}
        dict_info = form_dict(training_set,training_label)
        mean_dict=find_mean(dict_info)
        std_dict = find_std (dict_info,mean_dict)
        prior=find_priors(dict_info)





        _,predictions0 = fun_bayes(training_set,test_set,training_label,test_label)

        acc_funbayes+= _
        micro_funbayes+=f1_score(test_label,predictions0,average='micro')
        macro_funbayes+=f1_score(test_label,predictions0,average='macro')

        #print(macro_funbayes)



        _,predictions1=fun_knn(training_set,test_set,training_label,test_label)
        acc_funknn+= _
        micro_funknn+=f1_score(test_label,predictions1,average='micro')
        macro_funknn+=f1_score(test_label,predictions1,average='macro')




        _,predictions2=knn(training_set,training_label,test_set,test_label)
        acc_knn+= _


        micro_knn+=f1_score(test_label,predictions2,average='micro')
        macro_knn+=f1_score(test_label,predictions2,average='macro')



        _,predictions3=bayes(mean_dict,std_dict,test_set,test_label,prior)
        acc_bayes+= _
        micro_bayes+=f1_score(test_label,predictions3,average='micro')
        macro_bayes+=f1_score(test_label,predictions3,average='macro')



    file1.write("Test Accuracy on dolphin using inbuilt bayes  ::" + str(acc_funbayes/5) +"\n")
    file1.write("Test Accuracy on dolphin using inbuilt knn ::" + str(acc_funknn/5) +"\n")
    file1.write("Test Accuracy on dolphin using my bayes ::" + str(acc_bayes/5) +"\n")
    file1.write("Test Accuracy on dolphin using my knn ::" + str(acc_knn/5) +"\n \n")

    file1.write("Test Macro F1 Score on dolphin using inbuilt bayes ::" + str(macro_funbayes/5) +"\n")
    file1.write("Test Macro F1 Score on dolphin using inbuilt knn ::" + str(macro_funknn/5) +"\n")
    file1.write("Test Macro F1 Score on dolphin using my bayes ::" + str(macro_bayes/5) +"\n")
    file1.write("Test Macro F1 Score on dolphin using my knn ::" + str(macro_knn/5) +"\n \n")

    file1.write("Test Micro F1 Score on dolphin using inbuilt bayes ::" + str(micro_funbayes/5) +"\n")
    file1.write("Test Micro F1 Score on dolphin using inbuilt knn ::" + str(micro_funknn/5) +"\n")
    file1.write("Test Micro F1 Score on dolphin using my bayes ::" + str(micro_bayes/5) +"\n")
    file1.write("Test Micro F1 Score on dolphin using my knn ::" + str(micro_knn/5) +"\n \n")

    print(acc_bayes/5)
    print(acc_funbayes/5)
    print(acc_funknn/5)
    print(acc_knn/5)
    print(micro_bayes/5)
    print(micro_funbayes/5)
    print(micro_funknn/5)
    print(micro_knn/5)
    print(macro_bayes/5)
    print(macro_funbayes/5)
    print(macro_funknn/5)
    print(macro_knn/5)
    #file1.close()

    # ''''''''''''''''''''''''''''''''''''''''''''''''' for pubmed '''''''''''''''''''''''''''''''''''''''''''''
    #print("done")

def pubmed():
    train=load_data("E:\\pubmed.csv")
    m,n = train.shape
    labels=load_data("E:\\pubmed_label.csv")
    train_copy=train
    list_k_fold= man_split(train_copy,labels,5)
    acc_bayes=0
    acc_funbayes=0
    acc_knn=0
    acc_funknn=0

    #print("done")

    micro_bayes=0
    micro_funbayes=0
    micro_knn=0
    micro_funknn=0
    macro_bayes=0
    macro_funbayes=0
    macro_knn=0
    macro_funknn=0
    for k1 in range(5):
        #print(k1)
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

        #print("done")

        for i2 in range(len(labels)):
            if i2 in label1:
                test_set.append(train_copy[i2])
                test_label.append(labels[i2])
            else:
                training_set.append(train_copy[i2])
                #print(trainset)
                training_label.append(labels[i2])

        #print("done")

        dict_info={}
        dict_info = form_dict(training_set,training_label)
        mean_dict=find_mean(dict_info)
        std_dict = find_std (dict_info,mean_dict)
        prior=find_priors(dict_info)





        _,predictions0 = fun_bayes(training_set,test_set,training_label,test_label)

        acc_funbayes+= _
        micro_funbayes+=f1_score(test_label,predictions0,average='micro')
        macro_funbayes+=f1_score(test_label,predictions0,average='macro')

        #print(macro_funbayes)
        #print("a")


        _,predictions1=fun_knn(training_set,test_set,training_label,test_label)
        acc_funknn+= _
        micro_funknn+=f1_score(test_label,predictions1,average='micro')
        macro_funknn+=f1_score(test_label,predictions1,average='macro')
        #print("b")



        _,predictions2=knn(training_set,training_label,test_set,test_label)
        acc_knn+= _


        micro_knn+=f1_score(test_label,predictions2,average='micro')
        macro_knn+=f1_score(test_label,predictions2,average='macro')



        _,predictions3=bayes(mean_dict,std_dict,test_set,test_label,prior)
        acc_bayes+= _
        micro_bayes+=f1_score(test_label,predictions3,average='micro')
        macro_bayes+=f1_score(test_label,predictions3,average='macro')
        #print("c")


    file1.write("Test Accuracy on pubmed using inbuilt bayes  ::" + str(acc_funbayes/5) +"\n")
    file1.write("Test Accuracy on pubmed using inbuilt knn ::" + str(acc_funknn/5) +"\n")
    file1.write("Test Accuracy on pubmed using my bayes ::" + str(acc_bayes/5) +"\n")
    file1.write("Test Accuracy on pubmed using my knn ::" + str(acc_knn/5) +"\n \n")

    file1.write("Test Macro F1 Score on pubmed using inbuilt bayes ::" + str(macro_funbayes/5) +"\n")
    file1.write("Test Macro F1 Score on pubmed using inbuilt knn ::" + str(macro_funknn/5) +"\n")
    file1.write("Test Macro F1 Score on pubmed using my bayes ::" + str(macro_bayes/5) +"\n")
    file1.write("Test Macro F1 Score on pubmed using my knn ::" + str(macro_knn/5) +"\n \n")

    file1.write("Test Micro F1 Score on pubmed using inbuilt bayes ::" + str(micro_funbayes/5) +"\n")
    file1.write("Test Micro F1 Score on pubmed using inbuilt knn ::" + str(micro_funknn/5) +"\n")
    file1.write("Test Micro F1 Score on pubmed using my bayes ::" + str(micro_bayes/5) +"\n")
    file1.write("Test Micro F1 Score on pubmed using my knn ::" + str(micro_knn/5) +"\n \n")

    print(acc_bayes/5)
    print(acc_funbayes/5)
    print(acc_funknn/5)
    print(acc_knn/5)
    print(micro_bayes/5)
    print(micro_funbayes/5)
    print(micro_funknn/5)
    print(micro_knn/5)
    print(macro_bayes/5)
    print(macro_funbayes/5)
    print(macro_funknn/5)
    print(macro_knn/5)
    #file1.close()


    #''''''''''''''''''''''''''''''''''''''for twitter'''''''''''''''''''''''''''''''''''''''''




def twitter():
    global best
    global second_best
    train=process_twitter()

    m,n = train.shape
    labels=load_data("E:\\twitter_label.txt")
    train_copy=train
    list_k_fold= man_split(train_copy,labels,5)
    acc_bayes=0
    acc_funbayes=0
    acc_knn=0
    acc_funknn=0

    #print("done")

    micro_bayes=0
    micro_funbayes=0
    micro_knn=0
    micro_funknn=0
    macro_bayes=0
    macro_funbayes=0
    macro_knn=0
    macro_funknn=0
    for k1 in range(5):
        #print(k1)
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

        #print("done")

        for i2 in range(len(labels)):
            if i2 in label1:
                test_set.append(train_copy[i2])
                test_label.append(labels[i2])
            else:
                training_set.append(train_copy[i2])
                #print(trainset)
                training_label.append(labels[i2])

        #print("done")

        dict_info={}
        dict_info = form_dict(training_set,training_label)
        mean_dict=find_mean(dict_info)
        std_dict = find_std (dict_info,mean_dict)
        prior=find_priors(dict_info)





        _,predictions0 = fun_bayes(training_set,test_set,training_label,test_label)

        acc_funbayes+= _
        micro_funbayes+=f1_score(test_label,predictions0,average='micro')
        macro_funbayes+=f1_score(test_label,predictions0,average='macro')

        print(macro_funbayes)



        _,predictions1=fun_knn(training_set,test_set,training_label,test_label)
        acc_funknn+= _
        micro_funknn+=f1_score(test_label,predictions1,average='micro')
        macro_funknn+=f1_score(test_label,predictions1,average='macro')




        _,predictions2=knn(training_set,training_label,test_set,test_label)
        acc_knn+= _


        micro_knn+=f1_score(test_label,predictions2,average='micro')
        macro_knn+=f1_score(test_label,predictions2,average='macro')



        _,predictions3=bayes(mean_dict,std_dict,test_set,test_label,prior)
        acc_bayes+= _
        micro_bayes+=f1_score(test_label,predictions3,average='micro')
        macro_bayes+=f1_score(test_label,predictions3,average='macro')



    file1.write("Test Accuracy on twitter using inbuilt bayes  ::" + str(acc_funbayes/5) +"\n")
    file1.write("Test Accuracy on twitter using inbuilt knn ::" + str(acc_funknn/5) +"\n")
    file1.write("Test Accuracy on twitter using my bayes ::" + str(acc_bayes/5) +"\n")
    file1.write("Test Accuracy on twitter using my knn ::" + str(acc_knn/5) +"\n \n")

    file1.write("Test Macro F1 Score on twitter using inbuilt bayes ::" + str(macro_funbayes/5) +"\n")
    file1.write("Test Macro F1 Score on twitter using inbuilt knn ::" + str(macro_funknn/5) +"\n")
    file1.write("Test Macro F1 Score on twitter using my bayes ::" + str(macro_bayes/5) +"\n")
    file1.write("Test Macro F1 Score on twitter using my knn ::" + str(macro_knn/5) +"\n \n")

    file1.write("Test Micro F1 Score on twitter using inbuilt bayes ::" + str(micro_funbayes/5) +"\n")
    file1.write("Test Micro F1 Score on twitter using inbuilt knn ::" + str(micro_funknn/5) +"\n")
    file1.write("Test Micro F1 Score on twitter using my bayes ::" + str(micro_bayes/5) +"\n")
    file1.write("Test Micro F1 Score on twitter using my knn ::" + str(micro_knn/5) +"\n \n")
    file1.close()
    print(acc_bayes/5)
    print(acc_funbayes/5)
    print(acc_funknn/5)
    print(acc_knn/5)
    print(micro_bayes/5)
    print(micro_funbayes/5)
    print(micro_funknn/5)
    print(micro_knn/5)
    print(macro_bayes/5)
    print(macro_funbayes/5)
    print(macro_funknn/5)
    print(macro_knn/5)
    #print(best)
    #print(second_best)



#start()
#twitter()
#whole_dataset()
#pubmed()
##X=load_data("E:\\dolphins.csv")
#components=int(X.shape[1]//10)
#Z=pca(X,components)
#print(np.shape(Z))
#print(np.shape(X))




#knn(B,L,B)

#print(A)





#twitter()

#fun_bayes()

#fun_bayes()

#fun_knn ()
#A=[0.11850819578057319, 0.05443445414975332, 0.07477523401761366, 0.050437466919702086, 0.04943645778582566, 0.07814113173654262, 0.053692843981608326, 0.044370478807215924, 0.05215955751798727, 0.06049077052154263, 0.062410670126500714, 0.14629013899880586, 0.08446034254656638, 0.047174631581155015, 0.05170373735670413, 0.09857839677707733, 0.07058579236212403, 0.1450861603381416, 0.09302101001676949, 0.09100143636774671, 0.08186351660461004, 0.03605918115394546, 0.1081439852124883, 0.07091001300396314, 0.03725243265264606, 0.04843438063444921, 0.06808519774139145, 0.041827769712702696, 0.06702090345057725, 0.05716624508226472, 0.055374023501908634, 0.05346351786180669]
#joint_likelihood(prior,mean_dict,std_dict,A)


#B=load_data("E:\\twitter.txt")
#projection(B,i)
#B=process_twitter ()
#L=load_data("E:\\twitter_label.txt")
#training_set,test_set,training_label,test_label = split_data(B,L)
#dict_info={}
#dict_info = form_dict(training_set,training_label)
#mean_dict=find_mean(dict_info)
#std_dict = find_std (dict_info,mean_dict)
#prior=find_priors(dict_info)



#bayes(mean_dict,std_dict,test_set,test_label,prior)


















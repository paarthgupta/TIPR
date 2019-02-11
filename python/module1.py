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
import datetime
import projections as p
import nn
import bayes as bay
import specially_for_twitter_smoothing as smooth
import lsh
import sys
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

curr=os.getcwd()
final=os.path.join(curr,r'output')
if not os.path.exists(final):
    eos.makedirs(final)

file=open(final + '/task3.txt','w')
file1=open(final + '/task4.txt','w')



def close_file():
    file.close()
    file1.close()

def accuracy(predictions,test_label):
    if(predictions[i]!=test_label[i]):
        count+=1
    accuracy=   ((len(test_set)-count)/len(test_set))*100
    #print("my_bayes")
    #print (accuracy)
    return accuracy,predictions
    #print (sup)
    return accuracy and predictions

def plot(x,y,k):
    #print(y)
    if k<4:
        if (k==1):
            a="dolphins_accuracy"
        elif (k==2):
            a="dolphins_microscore"
        else:
            a="dolphins_macroscore"
    elif k<7:
        if (k==4):
            a="pubmed_accuracy"
        elif (k==5):
            a="pubmed_microscore"
        else:
            a="pubmed_macroscore"

    elif k<10:
        if (k==7):
            a="twitter_accuracy"
        elif (k==8):
            a="twitter_microscore"
        else:
            a="twitter_macroscore"
    #print (y)
    for i in range (len(y)):

        plt.plot(x, y[i], label="")
        #plt.plot(x, x, label='dolphin')
    plt.legend(['my_bayes','inbuilt_bayes','inbuilt_knn','my_knn'])
    plt.xlabel('Values of D')
    plt.ylabel(a)
    plt.savefig(''+str(a)+'.png')
    #plt.show()
    plt.clf()

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





def process_twitter(testpath):
    X2 = pd.read_csv("E:\\twitter.txt",header=None)
    test_set=pd.read_csv(testsetpath,header=None)
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

    all_words = set()
    local_ls = []
    for indx,stmt in test_set.iterrows():
        local = {}
        for word in stmt[0].strip().split():
            if word not in stopwords.words('english') :
                if word in local:
                    local[word] += 1
                else:
                    local[word]  = 1
        local_ls.append(local)
        all_words.update(local)
    mat1 = [[(local[word] if word in local else 0) for word in all_words] for local in local_ls]


    X2=np.array(mat)
    Y2=np.array(mat1)

    return(X2,Y2)

if __name__ =='__main__':
    args=sys.argv
    dataset_names = ['dolphins','pubmed','twitter']
    testsetpath=args[2]
    testsetlabel=args[4]

    given_dataset_name = args[6]
    i = dataset_names.index(given_dataset_name)

    if (i==0):

        d=2

        train=load_data("E:\\dolphins.csv")
        test_set=np.array(load_data(testsetpath))
        test_label=np.array(load_data(testsetlabel))
        train_lsh=np.array(train)
        training_set=train_lsh
        test_set_lsh=test_set

        m,n = train.shape
        labels=load_data("E:\\dolphins_label.csv")
        label_lsh=np.array(labels)
        training_label=label_lsh

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
            train_copy=p.projection(train,d,1)
            training_set=train_copy
            test_copy=p.projection(test_set,d,1)
            test_set=test_copy
            #training_set,test_set,training_label,test_label
            #list_k_fold= man_split(train_copy,labels,5)
            #k=5
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


            dict_info={}
            dict_info = form_dict(training_set,training_label)
            mean_dict=bay.find_mean(dict_info)
            std_dict = bay.find_std (dict_info,mean_dict)
            prior=bay.find_priors(dict_info)

                #print(len(training_set))
                #print(len(test_set))
            _,predictions0 = bay.fun_bayes(training_set,test_set,training_label,test_label)
            acc_funbayes+= _
            micro_funbayes+=f1_score(test_label,predictions0,average='micro')
            macro_funbayes+=f1_score(test_label,predictions0,average='macro')


            _,predictions1=nn.fun_knn(training_set,test_set,training_label,test_label)
            acc_funknn+= _
            micro_funknn+=f1_score(test_label,predictions1,average='micro')
            macro_funknn+=f1_score(test_label,predictions1,average='macro')


            _,predictions2=nn.knn(training_set,training_label,test_set,test_label)
            acc_knn+= _
            micro_knn+=f1_score(test_label,predictions2,average='micro')
            macro_knn+=f1_score(test_label,predictions2,average='macro')


            _,predictions3=bay.bayes(mean_dict,std_dict,test_set,test_label,prior)
            acc_bayes+= _
            micro_bayes+=f1_score(test_label,predictions3,average='micro')
            macro_bayes+=f1_score(test_label,predictions3,average='macro')

            file1.write("Test Accuracy on dolphin using inbuilt bayes and d =" + str(d) + "::" + str(acc_funbayes) +"\n")
            file1.write("Test Accuracy on dolphin using inbuilt knn and d =" + str(d) + "::" + str(acc_funknn) +"\n")
            file.write("Test Accuracy on dolphin using my bayes and d =" + str(d) + "::" + str(acc_bayes) +"\n")
            file.write("Test Accuracy on dolphin using my knn and d =" + str(d) + "::" + str(acc_knn) +"\n \n")

            file1.write("Test Macro F1 Score on dolphin using inbuilt bayes and d =" + str(d) + "::" + str(macro_funbayes) +"\n")
            file1.write("Test Macro F1 Score on dolphin using inbuilt knn and d =" + str(d) + "::" + str(macro_funknn) +"\n")
            file.write("Test Macro F1 Score on dolphin using my bayes and d =" + str(d) + "::" + str(macro_bayes) +"\n")
            file.write("Test Macro F1 Score on dolphin using my knn and d =" + str(d) + "::" + str(macro_knn) +"\n \n")

            file1.write("Test Micro F1 Score on dolphin using inbuilt bayes and d =" + str(d) + "::" + str(micro_funbayes) +"\n")
            file1.write("Test Micro F1 Score on dolphin using inbuilt knn and d =" + str(d) + "::" + str(micro_funknn) +"\n")
            file.write("Test Micro F1 Score on dolphin using my bayes and d =" + str(d) + "::" + str(micro_bayes) +"\n")
            file.write("Test Micro F1 Score on dolphin using my knn and d =" + str(d) + "::" + str(micro_knn) +"\n \n")




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
            list1.append(acc_bayes)

            list2.append(acc_funbayes)
            list3.append(acc_funknn)
            list4.append(acc_knn)
            list5.append(micro_bayes)
            list6.append(micro_funbayes)
            list7.append(micro_funknn)
            list8.append(micro_knn)
            list9.append(macro_bayes)
            list10.append(macro_funbayes)
            list11.append(macro_funknn)
            list12.append(macro_knn)


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
        lsh.start(train_lsh,label_lsh,test_set_lsh,test_label,1)

    elif(i==1):

        d=2
        list1=[]
        lidt2=[]
        list3=[]

        train=load_data("E:\\pubmed.csv")
        train_lsh=np.array(train)
        training_set=train_lsh
        test_set=np.array(load_data(testsetpath))
        test_label=np.array(load_data(testsetlabel))

        m,n = train.shape
        labels=load_data("E:\\pubmed_label.csv")
        label_lsh=np.array(labels)
        training_label=label_lsh
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
            #print(d)
            #train_copy=train
            train_copy=p.projection(train,d,2)

            training_set=train_copy
            test_copy=p.projection(test_set,d,2)
            test_set=test_copy


            #training_set,test_set,training_label,test_label
            #list_k_fold= man_split(train_copy,labels,5)
            #k=5



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


            dict_info={}
            dict_info = form_dict(training_set,training_label)
            mean_dict=bay.find_mean(dict_info)
            std_dict = bay.find_std (dict_info,mean_dict)
            prior=bay.find_priors(dict_info)

            _,predictions0=bay.fun_bayes(training_set,test_set,training_label,test_label)
            acc_funbayes+= _
            micro_funbayes+=f1_score(test_label,predictions0,average='micro')
            macro_funbayes+=f1_score(test_label,predictions0,average='macro')

            #print("1")

            _,predictions1=nn.fun_knn(training_set,test_set,training_label,test_label)
            acc_funknn+= _
            micro_funknn+=f1_score(test_label,predictions1,average='micro')
            macro_funknn+=f1_score(test_label,predictions1,average='macro')

            #print("2")
            _,predictions2=nn.knn(training_set,training_label,test_set,test_label)
            acc_knn+= _
            micro_knn+=f1_score(test_label,predictions2,average='micro')
            macro_knn+=f1_score(test_label,predictions2,average='macro')

            #print("3")
            _,predictions3=bay.bayes(mean_dict,std_dict,test_set,test_label,prior)
            acc_bayes+= _
            micro_bayes+=f1_score(test_label,predictions3,average='micro')
            macro_bayes+=f1_score(test_label,predictions3,average='macro')

                #print("4")
            file1.write("Test Accuracy on pubmed using inbuilt bayes and d =" + str(d) + "::" + str(acc_funbayes) +"\n")
            file1.write("Test Accuracy on pubmed using inbuilt knn and d =" + str(d) + "::" + str(acc_funknn) +"\n")
            file.write("Test Accuracy on pubmed using my bayes and d =" + str(d) + "::" + str(acc_bayes) +"\n")
            file.write("Test Accuracy on pubmed using my knn and d =" + str(d) + "::" + str(acc_knn) +"\n \n")

            file1.write("Test Macro F1 Score on pubmed using inbuilt bayes and d =" + str(d) + "::" + str(macro_funbayes) +"\n")
            file1.write("Test Macro F1 Score on pubmed using inbuilt knn and d =" + str(d) + "::" + str(macro_funknn) +"\n")
            file.write("Test Macro F1 Score on pubmed using my bayes and d =" + str(d) + "::" + str(macro_bayes) +"\n")
            file.write("Test Macro F1 Score on pubmed using my knn and d =" + str(d) + "::" + str(macro_knn) +"\n \n")

            file1.write("Test Micro F1 Score on pubmed using inbuilt bayes and d =" + str(d) + "::" + str(micro_funbayes) +"\n")
            file1.write("Test Micro F1 Score on pubmed using inbuilt knn and d =" + str(d) + "::" + str(micro_funknn) +"\n")
            file.write("Test Micro F1 Score on pubmed using my bayes and d =" + str(d) + "::" + str(micro_bayes) +"\n")
            file.write("Test Micro F1 Score on pubmed using my knn and d =" + str(d) + "::" + str(micro_knn) +"\n \n")


#                print(acc_bayes/5)
#               print(acc_funbayes/5)
#              print(acc_funknn/5)
#             print(acc_knn/5)
#            print(micro_bayes/5)
 #           print(micro_funbayes/5)
  #          print(micro_funknn/5)
   #         print(micro_knn/5)
    #        print(macro_bayes/5)
     #       print(macro_funbayes/5)
      #      print(macro_funknn/5)
       #     print(macro_knn/5)

            x.append(d)
            list1.append(acc_bayes)
            #print(acc_bayes/5)
            list2.append(acc_funbayes)
            #print(acc_funbayes/5)
            list3.append(acc_funknn)
            list4.append(acc_knn)
            list5.append(micro_bayes)
            list6.append(micro_funbayes)
            list7.append(micro_funknn)
            list8.append(micro_knn)
            list9.append(macro_bayes)
            list10.append(macro_funbayes)
            list11.append(macro_funknn)
            list12.append(macro_knn)


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
        lsh.start(train_lsh,label_lsh,test_set,test_label,2)


    elif (i==2) :
        d=2
        list1=[]
        lidt2=[]
        list3=[]
        train,test_set=process_twitter(testsetpath)
        train_lsh=np.array(train)
        training_set=train_lsh

        test_label=np.array(load_data(testsetlabel))


        m,n = train.shape
        #print(end)
        #print(B)
        #projection(B,i)
        labels=load_data("E:\\twitter_label.txt")
        label_lsh=np.array(labels)
        training_label=label_lsh
        #test_set=process_twitter(test_set)
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
            print("tw")
            train_copy=train
            training_set=train_copy
            training_set=p.projection(training_set,d,3)
            test_copy=p.projection(test_set,d,3)
            test_set=test_copy
            #train_copy=projection(train,d,3)



            #list_k_fold= man_split(train_copy,labels,5)
            #k=5
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


            dict_info={}
            dict_info = form_dict(training_set,training_label)
            mean_dict=bay.find_mean(dict_info)
            std_dict = bay.find_std (dict_info,mean_dict)
            prior=bay.find_priors(dict_info)

            #print(len(training_set))
            #print(len(test_set))

            _,predictions0=bay.fun_bayes(training_set,test_set,training_label,test_label)
            acc_funbayes+= _
            micro_funbayes+=f1_score(test_label,predictions0,average='micro')
            macro_funbayes+=f1_score(test_label,predictions0,average='macro')

            print(micro_funbayes)

            _,predictions1=nn.fun_knn(training_set,test_set,training_label,test_label)
            acc_funknn+= _
            micro_funknn+=f1_score(test_label,predictions1,average='micro')
            macro_funknn+=f1_score(test_label,predictions1,average='macro')

            print(micro_funknn)

            _,predictions2=nn.knn(training_set,training_label,test_set,test_label)
            acc_knn+= _
            micro_knn+=f1_score(test_label,predictions2,average='micro')
            macro_knn+=f1_score(test_label,predictions2,average='macro')

            #print(micro_bayes)

            _,predictions3=bay.bayes(mean_dict,std_dict,test_set,test_label,prior)
            acc_bayes+= _
            micro_bayes+=f1_score(test_label,predictions3,average='micro')
            macro_bayes+=f1_score(test_label,predictions3,average='macro')

                #print(micro_bayes/5)





            file1.write("Test Accuracy on twitter using inbuilt bayes and d =" + str(d) + "::" + str(acc_funbayes) +"\n")
            file1.write("Test Accuracy on twitter using inbuilt knn and d =" + str(d) + "::" + str(acc_funknn) +"\n")
            file.write("Test Accuracy on twitter using my bayes and d =" + str(d) + "::" + str(acc_bayes) +"\n")
            file.write("Test Accuracy on twitter using my knn and d =" + str(d) + "::" + str(acc_knn) +"\n \n")

            file1.write("Test Macro F1 Score on twitter using inbuilt bayes and d =" + str(d) + "::" + str(macro_funbayes) +"\n")
            file1.write("Test Macro F1 Score on twitter using inbuilt knn and d =" + str(d) + "::" + str(macro_funknn) +"\n")
            file.write("Test Macro F1 Score on twitter using my bayes and d =" + str(d) + "::" + str(macro_bayes) +"\n")
            file.write("Test Macro F1 Score on twitter using my knn and d =" + str(d) + "::" + str(macro_knn) +"\n \n")

            file1.write("Test Micro F1 Score on twitter using inbuilt bayes and d =" + str(d) + "::" + str(micro_funbayes) +"\n")
            file1.write("Test Micro F1 Score on twitter using inbuilt knn and d =" + str(d) + "::" + str(micro_funknn) +"\n")
            file.write("Test Micro F1 Score on twitter using my bayes and d =" + str(d) + "::" + str(micro_bayes) +"\n")
            file.write("Test Micro F1 Score on twitter using my knn and d =" + str(d) + "::" + str(micro_knn) +"\n \n")

#                print(acc_bayes/5)
#               print(acc_funbayes/5)
#              print(acc_funknn/5)
#             print(acc_knn/5)
#            print(micro_bayes/5)
 #           print(micro_funbayes/5)
  #          print(micro_funknn/5)
   #         print(micro_knn/5)
    #        print(macro_bayes/5)
     #       print(macro_funbayes/5)
      #      print(macro_funknn/5)
       #     print(macro_knn/5)
            x.append(d)
            list1.append(acc_bayes)
            #print(acc_bayes/5)
            list2.append(acc_funbayes)
            #print(acc_funbayes/5)
            list3.append(acc_funknn)
            list4.append(acc_knn)
            list5.append(micro_bayes)
            list6.append(micro_funbayes)
            list7.append(micro_funknn)
            list8.append(micro_knn)
            list9.append(macro_bayes)
            list10.append(macro_funbayes)
            list11.append(macro_funknn)
            list12.append(macro_knn)

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
        lsh.start(train_lsh,label_lsh,3)
        #file.close()

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
        mean_dict=bay.find_mean(dict_info)
        std_dict = bay.find_std (dict_info,mean_dict)
        prior=bay.find_priors(dict_info)





        _,predictions0 = bay.fun_bayes(training_set,test_set,training_label,test_label)

        acc_funbayes+= _
        micro_funbayes+=f1_score(test_label,predictions0,average='micro')
        macro_funbayes+=f1_score(test_label,predictions0,average='macro')

        #print(macro_funbayes)



        _,predictions1=nn.fun_knn(training_set,test_set,training_label,test_label)
        acc_funknn+= _
        micro_funknn+=f1_score(test_label,predictions1,average='micro')
        macro_funknn+=f1_score(test_label,predictions1,average='macro')




        _,predictions2=nn.knn(training_set,training_label,test_set,test_label)
        acc_knn+= _


        micro_knn+=f1_score(test_label,predictions2,average='micro')
        macro_knn+=f1_score(test_label,predictions2,average='macro')



        _,predictions3=bay.bayes(mean_dict,std_dict,test_set,test_label,prior)
        acc_bayes+= _
        micro_bayes+=f1_score(test_label,predictions3,average='micro')
        macro_bayes+=f1_score(test_label,predictions3,average='macro')



    file1.write("Test Accuracy on dolphin using inbuilt bayes  ::" + str(acc_funbayes/5) +"\n")
    file1.write("Test Accuracy on dolphin using inbuilt knn ::" + str(acc_funknn/5) +"\n")
    file.write("Test Accuracy on dolphin using my bayes ::" + str(acc_bayes/5) +"\n")
    file.write("Test Accuracy on dolphin using my knn ::" + str(acc_knn/5) +"\n \n")

    file1.write("Test Macro F1 Score on dolphin using inbuilt bayes ::" + str(macro_funbayes/5) +"\n")
    file1.write("Test Macro F1 Score on dolphin using inbuilt knn ::" + str(macro_funknn/5) +"\n")
    file.write("Test Macro F1 Score on dolphin using my bayes ::" + str(macro_bayes/5) +"\n")
    file.write("Test Macro F1 Score on dolphin using my knn ::" + str(macro_knn/5) +"\n \n")

    file1.write("Test Micro F1 Score on dolphin using inbuilt bayes ::" + str(micro_funbayes/5) +"\n")
    file1.write("Test Micro F1 Score on dolphin using inbuilt knn ::" + str(micro_funknn/5) +"\n")
    file.write("Test Micro F1 Score on dolphin using my bayes ::" + str(micro_bayes/5) +"\n")
    file.write("Test Micro F1 Score on dolphin using my knn ::" + str(micro_knn/5) +"\n \n")

 #   print(acc_bayes/5)
  #  print(acc_funbayes/5)
   # print(acc_funknn/5)
    #print(acc_knn/5)
#    print(micro_bayes/5)
 #   print(micro_funbayes/5)
  #  print(micro_funknn/5)
   # print(micro_knn/5)
    #print(macro_bayes/5)
#    print(macro_funbayes/5)
 #   print(macro_funknn/5)
  #  print(macro_knn/5)
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
        mean_dict=bay.find_mean(dict_info)
        std_dict = bay.find_std (dict_info,mean_dict)
        prior=bay.find_priors(dict_info)





        _,predictions0 = bay.fun_bayes(training_set,test_set,training_label,test_label)

        acc_funbayes+= _
        micro_funbayes+=f1_score(test_label,predictions0,average='micro')
        macro_funbayes+=f1_score(test_label,predictions0,average='macro')

        #print(macro_funbayes)
        #print("a")


        _,predictions1=nn.fun_knn(training_set,test_set,training_label,test_label)
        acc_funknn+= _
        micro_funknn+=f1_score(test_label,predictions1,average='micro')
        macro_funknn+=f1_score(test_label,predictions1,average='macro')
        #print("b")



        _,predictions2=nn.knn(training_set,training_label,test_set,test_label)
        acc_knn+= _


        micro_knn+=f1_score(test_label,predictions2,average='micro')
        macro_knn+=f1_score(test_label,predictions2,average='macro')



        _,predictions3=bay.bayes(mean_dict,std_dict,test_set,test_label,prior)
        acc_bayes+= _
        micro_bayes+=f1_score(test_label,predictions3,average='micro')
        macro_bayes+=f1_score(test_label,predictions3,average='macro')
        #print("c")


    file1.write("Test Accuracy on pubmed using inbuilt bayes  ::" + str(acc_funbayes/5) +"\n")
    file1.write("Test Accuracy on pubmed using inbuilt knn ::" + str(acc_funknn/5) +"\n")
    file.write("Test Accuracy on pubmed using my bayes ::" + str(acc_bayes/5) +"\n")
    file.write("Test Accuracy on pubmed using my knn ::" + str(acc_knn/5) +"\n \n")

    file1.write("Test Macro F1 Score on pubmed using inbuilt bayes ::" + str(macro_funbayes/5) +"\n")
    file1.write("Test Macro F1 Score on pubmed using inbuilt knn ::" + str(macro_funknn/5) +"\n")
    file.write("Test Macro F1 Score on pubmed using my bayes ::" + str(macro_bayes/5) +"\n")
    file.write("Test Macro F1 Score on pubmed using my knn ::" + str(macro_knn/5) +"\n \n")

    file1.write("Test Micro F1 Score on pubmed using inbuilt bayes ::" + str(micro_funbayes/5) +"\n")
    file1.write("Test Micro F1 Score on pubmed using inbuilt knn ::" + str(micro_funknn/5) +"\n")
    file.write("Test Micro F1 Score on pubmed using my bayes ::" + str(micro_bayes/5) +"\n")
    file.write("Test Micro F1 Score on pubmed using my knn ::" + str(micro_knn/5) +"\n \n")

#    print(acc_bayes/5)
 #   print(acc_funbayes/5)
  #  print(acc_funknn/5)
   # print(acc_knn/5)
    #print(micro_bayes/5)
#    print(micro_funbayes/5)
 #   print(micro_funknn/5)
  #  print(micro_knn/5)
   # print(macro_bayes/5)
    #print(macro_funbayes/5)
#    print(macro_funknn/5)
 #   print(macro_knn/5)
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
        mean_dict=bay.find_mean(dict_info)
        std_dict = bay.find_std (dict_info,mean_dict)
        prior=bay.find_priors(dict_info)





        _,predictions0 = bay.fun_bayes(training_set,test_set,training_label,test_label)

        acc_funbayes+= _
        micro_funbayes+=f1_score(test_label,predictions0,average='micro')
        macro_funbayes+=f1_score(test_label,predictions0,average='macro')

        #print(macro_funbayes)



        _,predictions1=nn.fun_knn(training_set,test_set,training_label,test_label)
        acc_funknn+= _
        micro_funknn+=f1_score(test_label,predictions1,average='micro')
        macro_funknn+=f1_score(test_label,predictions1,average='macro')




        _,predictions2=nn.knn(training_set,training_label,test_set,test_label)
        acc_knn+= _


        micro_knn+=f1_score(test_label,predictions2,average='micro')
        macro_knn+=f1_score(test_label,predictions2,average='macro')



        _,predictions3=bay.bayes(mean_dict,std_dict,test_set,test_label,prior)
        acc_bayes+= _
        micro_bayes+=f1_score(test_label,predictions3,average='micro')
        macro_bayes+=f1_score(test_label,predictions3,average='macro')



    file1.write("Test Accuracy on twitter using inbuilt bayes  ::" + str(acc_funbayes/5) +"\n")
    file1.write("Test Accuracy on twitter using inbuilt knn ::" + str(acc_funknn/5) +"\n")
    file.write("Test Accuracy on twitter using my bayes ::" + str(acc_bayes/5) +"\n")
    file.write("Test Accuracy on twitter using my knn ::" + str(acc_knn/5) +"\n \n")

    file1.write("Test Macro F1 Score on twitter using inbuilt bayes ::" + str(macro_funbayes/5) +"\n")
    file1.write("Test Macro F1 Score on twitter using inbuilt knn ::" + str(macro_funknn/5) +"\n")
    file.write("Test Macro F1 Score on twitter using my bayes ::" + str(macro_bayes/5) +"\n")
    file.write("Test Macro F1 Score on twitter using my knn ::" + str(macro_knn/5) +"\n \n")

    file1.write("Test Micro F1 Score on twitter using inbuilt bayes ::" + str(micro_funbayes/5) +"\n")
    file1.write("Test Micro F1 Score on twitter using inbuilt knn ::" + str(micro_funknn/5) +"\n")
    file.write("Test Micro F1 Score on twitter using my bayes ::" + str(micro_bayes/5) +"\n")
    file.write("Test Micro F1 Score on twitter using my knn ::" + str(micro_knn/5) +"\n \n")
    #file1.close()
#    print(acc_bayes/5)
 #   print(acc_funbayes/5)
  #  print(acc_funknn/5)
   # print(acc_knn/5)
    #print(micro_bayes/5)
#    print(micro_funbayes/5)
 #   print(micro_funknn/5)
  #  print(micro_knn/5)
   # print(macro_bayes/5)
    #print(macro_funbayes/5)
#    print(macro_funknn/5)
 #   print(macro_knn/5)
    smooth.start()
    #print(best)
    #print(second_best)
















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


















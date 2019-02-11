from copy import copy
from itertools import combinations
import numpy as np
from pandas import DataFrame
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import operator
#import prog1
import pandas as pd
from nltk.corpus import stopwords
from random import randrange
#from prog1.py import cross_validation_k,scikit_knn
from sklearn.neighbors import KNeighborsClassifier
import math
#import module1
import os
from sklearn.metrics import f1_score
import nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
    plt.legend(['lsh','pca'])
    plt.xlabel('Values of D')
    plt.ylabel(a)
    plt.savefig('pca vs lsh'+str(a)+'.png')
    #plt.show()
    plt.clf()

def pca (X,components):
    return pd.DataFrame(PCA(n_components=components).fit_transform(X))

def lsh_ac(training_set,dim,i):
    curr=os.getcwd()
    final=os.path.join(curr,r'LSH_files')
    if not os.path.exists(final):
        os.makedirs(final)

    np.random.seed(42)
    row,col = training_set.shape
    training_set=np.array(training_set)
    training_set=training_set-training_set.mean(axis=1,keepdims=True)
    training_set[training_set<=0]=0
    training_set[training_set>0]=1
    training_set=np.array(training_set,dtype=int)
    returning_set=pd.DataFrame(np.zeros((row,dim)))
    for i in range(dim):
        prn = np.random.permutation(col)+1
        new_x=training_set*prn
        new_x[new_x==0] = col+1
        returning_set[i]=np.array(new_x.min(axis=1),dtype=int)
    #f=open(final,'/task_6_dataset'+str(i)+'dimension'+str(dim)+'.txt','w')
   # f.write(str(np.array(returning_set)))
   # f.close()
    return np.array(pd.DataFrame(returning_set))

def hash_iter1(training_set,testing_set,nooftables):
    all_tables={}
    #print("-----------------")
   # print(len(training_set))
    dmns=len(training_set[0])
    uniform_var=[]
    rand_vec={}
    for j in range(nooftables):
        red_mat=np.random.standard_normal(1*dmns)
        red_mat=red_mat.reshape(1,dmns)
        rand_vec1 = np.asmatrix(red_mat)
        rand_vec[j]=np.array(rand_vec1).T
        reducedform=(np.array(training_set).dot(np.array(rand_vec1).T))
        all_tables_iter={}
        varb=np.random.uniform(0,5)
        uniform_var.append(varb)
        for i in range(len(reducedform)):
            L=reducedform[i]+varb
            iter1=(L/5)
            iter1=int(iter1)
            if iter1 not in all_tables_iter:
                all_tables_iter[iter1]=[]
            all_tables_iter[iter1].append(i)

        all_tables[j]=all_tables_iter
    #print(all_tables)
	#s=[0 for i1 in range(len(testing_set))]
    coll=[]
    for i in range(len(testing_set)):
        coll.append(0)
        coll[i]=set()
    for i in range(nooftables):
		#reducedform=np.array(testing_set).dot(np.array(rand_vec[i]))
        reducedform=(np.array(testing_set).dot(np.array(rand_vec[i])))
        for k1 in range(len(reducedform)):
            L=reducedform[k1]+uniform_var[i]
            iter1=(L/5)
            iter1=int(iter1)

            if iter1 in  all_tables[i]:
                coll[k1].update((all_tables[i][iter1]))
    #print(coll)
    return coll

def scikit_knn(testing_set,training_setset,testlabel,training_setlabel):
	#global testing_set,training_setset,training_setlabel,testlabel

	knn_classifier = KNeighborsClassifier(n_neighbors=(3 if len(training_setset)>=3 else len(training_setset)))
	#print(testlabel)
	knn_classifier.fit((training_setset),np.ravel(training_setlabel))

	predictions=knn_classifier.predict(testing_set)
	# print(len(predictions))
	# print("+++++++++++++++++++++++++++++++++++++")
	# counter=0
	# for i1 in range(len(predictions)):
	# 	if predictions[i1]!=testlabel[i1]:
	# 		counter=counter+1
	# #print(counter)
	# print("THE ACCURACY OF SKLEARN KNN CLASSIFIER IS :: %f" % (float((len(testing_set)-counter)/len(testing_set))*100))
	return predictions#,(float(len(testing_set)-counter)/len(testing_set)*100)


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



#find_dict=hash_iter1(training_setset,testing_set,20)



def start_lsh(find_dict,training_set,training_label,test_set,test_label):
    count=0
    #print(len(find_dict))
    pred=[]
    for i in range(len(find_dict)):
        list1=[]


        dictio=list(find_dict[i])

        for j in range(len(dictio)):

            list1.append(training_label[dictio[j]][0])

        form_dict={}
        #print("****************")
       # print(training_label[dictio[1]][0])
        for k in range(len(list1)):
            if list1[k] not in form_dict:
                form_dict[list1[k]]=0
            form_dict[list1[k]]+=1

        if len(dictio)==0:

            print("------------------------------------------------------------")
        else:
            #print("ooooooooo")
            list2=[]
            list3=[]
            for j in range(len(dictio)):
                list2.append(training_set[dictio[j]])
                list3.append(training_label[dictio[j]])

            l=max(form_dict.items(), key=operator.itemgetter(1))[0]
   #print(l,dic1,testlabel[i])
			# print(len(testing_set))
			# print(i)
			# print(testing_set[i])
            predictions=scikit_knn(np.array(test_set[i]).reshape(1,len(test_set[i])),list2,test_label[i],list3)
            pred.append(predictions)
            if predictions!=test_label[i][0]:
                count+=1

   # print("miss:")
    accuracy=float((len(test_set)-count)/len(test_set)*100)
    #print(len(test_set))
    #print(len(training_set))  ;
    #print(count)

    return accuracy,pred



def start(training_set11,training_label11,test_set,test_label,ii):
    curr=os.getcwd()
    final=os.path.join(curr,r'LSH_files')
    if not os.path.exists(final):
        os.makedirs(final)


    file4=open(final + '/task7'+str(ii)+'.txt','w')


    n,m=training_set11.shape
    i3=2

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
    while(i3<=m):
        #print(i3,end=" ")
       # list_k_fold=man_split(training_set11,training_label11,5)
        train_copy=lsh_ac(training_set11,i3,ii)
        test_copy=lsh_ac(test_set,i3,ii)
        train_copy_pca=np.array(pca(training_set11,i3))
        test_copy_pca=np.array(pca(test_set,i3))

        #print(len(train_copy))

        accuracy=0
        micro=0
        macro=0
        accuracy_pca=0
        micro_pca=0
        macro_pca=0


            #print("-------------")
            #print(len(training_set))
        find_dict=hash_iter1(train_copy,test_copy,20)
        _,predictions=start_lsh(find_dict,train_copy,training_label11,test_copy,test_label)
        #print(predictions)
        accuracy+= _
        micro+=f1_score(test_label,predictions,average='micro')
        macro+=f1_score(test_label,predictions,average='macro')

        _,pred1=nn.fun_knn(train_copy_pca,test_copy_pca,training_label11,test_label)
        accuracy_pca+=_
        micro_pca+=f1_score(test_label,pred1,average='micro')
        macro_pca+=f1_score(test_label,pred1,average='macro')

        x.append(i3)
        list1.append(accuracy)

        list2.append(macro)
        list3.append(micro)
        list4.append(accuracy_pca)
        list5.append(macro_pca)
        list6.append(micro_pca)

        file4.write("Test Accuracy on dataset " + str(ii)+ " using LSH on d = "+str(i3)+"="+str(accuracy) +"\n")
        file4.write("Test Accuracy on dataset " + str(ii)+ " using PCA on d = "+str(i3)+"="+str(accuracy_pca) +"\n")
        file4.write("Macro Score on dataset " + str(ii)+ " using LSH on d = "+str(i3)+"="+str(macro) +"\n")
        file4.write("Macro Score on dataset " + str(ii)+ " using PCA on d = "+str(i3)+"="+str(macro_pca) +"\n")
        file4.write("Micro Score on dataset " + str(ii)+ " using LSH on d = "+str(i3)+"="+str(micro) +"\n")
        file4.write("Micro Score on dataset " + str(ii)+ " using PCA on d = "+str(i3)+"="+str(micro_pca) +"\n")
        i3=i3*2
        print(accuracy)
#       print(micro/5)
#      print(macro/5)
#     print(accuracy_pca/5)
#    print(micro_pca/5)
 #   print(macro_pca/5)
    a.append(list1)
    a.append(list4)
    b.append(list2)
    b.append(list5)
    c.append(list3)
    c.append(list6)
    plot(x,a,1+(3*(ii-1)))
    plot(x,b,2+(3*(ii-1)))
    plot(x,c,3+(3*(ii-1)))

def abc():
    print("FOR pubmed DATA SET: ")
    training_set = pd.read_csv("E:\\dolphins.csv",delimiter=' ',header=None)
    labels=pd.read_csv("E:\\dolphins_label.csv",delimiter=' ',header=None)
    test_set = pd.read_csv("E:\\dolphins.csv",delimiter=' ',header=None)
    test_labels=pd.read_csv("E:\\dolphins_label.csv",delimiter=' ',header=None)

    training_set=np.array(training_set)
    test_set=np.array(test_set)
    test_label=np.array(test_labels)
    labels=np.array(labels)
    n,m=training_set.shape
    print(n,m)
    start(training_set,labels,test_set,test_label,1)








#abc()


















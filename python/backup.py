def start():

    for i in range(3):
       # dataset_names = ['dolphin','pubmed','twitter']
      #  given_dataset_name = sys.argv[6]
      #  i = dataset_names.index(given_dataset_name)

        if (i==0):

            d=2

            train=load_data("E:\\dolphins.csv")
            train_lsh=np.array(train)

            m,n = train.shape
            labels=load_data("E:\\dolphins_label.csv")
            label_lsh=np.array(labels)
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
                train_copy=p.projection(train,d,1)
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

                file1.write("Test Accuracy on dolphin using inbuilt bayes and d =" + str(d) + "::" + str(acc_funbayes/5) +"\n")
                file1.write("Test Accuracy on dolphin using inbuilt knn and d =" + str(d) + "::" + str(acc_funknn/5) +"\n")
                file.write("Test Accuracy on dolphin using my bayes and d =" + str(d) + "::" + str(acc_bayes/5) +"\n")
                file.write("Test Accuracy on dolphin using my knn and d =" + str(d) + "::" + str(acc_knn/5) +"\n \n")

                file1.write("Test Macro F1 Score on dolphin using inbuilt bayes and d =" + str(d) + "::" + str(macro_funbayes/5) +"\n")
                file1.write("Test Macro F1 Score on dolphin using inbuilt knn and d =" + str(d) + "::" + str(macro_funknn/5) +"\n")
                file.write("Test Macro F1 Score on dolphin using my bayes and d =" + str(d) + "::" + str(macro_bayes/5) +"\n")
                file.write("Test Macro F1 Score on dolphin using my knn and d =" + str(d) + "::" + str(macro_knn/5) +"\n \n")

                file1.write("Test Micro F1 Score on dolphin using inbuilt bayes and d =" + str(d) + "::" + str(micro_funbayes/5) +"\n")
                file1.write("Test Micro F1 Score on dolphin using inbuilt knn and d =" + str(d) + "::" + str(micro_funknn/5) +"\n")
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
            lsh.start(train_lsh,label_lsh,1)

        elif(i==1):

            d=2
            list1=[]
            lidt2=[]
            list3=[]

            train=load_data("E:\\pubmed.csv")
            train_lsh=np.array(train)

            m,n = train.shape
            labels=load_data("E:\\pubmed_label.csv")
            train_label=np.array(labels)
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
                file1.write("Test Accuracy on pubmed using inbuilt bayes and d =" + str(d) + "::" + str(acc_funbayes/5) +"\n")
                file1.write("Test Accuracy on pubmed using inbuilt knn and d =" + str(d) + "::" + str(acc_funknn/5) +"\n")
                file.write("Test Accuracy on pubmed using my bayes and d =" + str(d) + "::" + str(acc_bayes/5) +"\n")
                file.write("Test Accuracy on pubmed using my knn and d =" + str(d) + "::" + str(acc_knn/5) +"\n \n")

                file1.write("Test Macro F1 Score on pubmed using inbuilt bayes and d =" + str(d) + "::" + str(macro_funbayes/5) +"\n")
                file1.write("Test Macro F1 Score on pubmed using inbuilt knn and d =" + str(d) + "::" + str(macro_funknn/5) +"\n")
                file.write("Test Macro F1 Score on pubmed using my bayes and d =" + str(d) + "::" + str(macro_bayes/5) +"\n")
                file.write("Test Macro F1 Score on pubmed using my knn and d =" + str(d) + "::" + str(macro_knn/5) +"\n \n")

                file1.write("Test Micro F1 Score on pubmed using inbuilt bayes and d =" + str(d) + "::" + str(micro_funbayes/5) +"\n")
                file1.write("Test Micro F1 Score on pubmed using inbuilt knn and d =" + str(d) + "::" + str(micro_funknn/5) +"\n")
                file.write("Test Micro F1 Score on pubmed using my bayes and d =" + str(d) + "::" + str(micro_bayes/5) +"\n")
                file.write("Test Micro F1 Score on pubmed using my knn and d =" + str(d) + "::" + str(micro_knn/5) +"\n \n")


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
            lsh.start(train_lsh,label_lsh,2)


        else :
            d=2
            list1=[]
            lidt2=[]
            list3=[]
            train=process_twitter()
            train_lsh=np.array(train)


            m,n = train.shape
            #print(end)
            #print(B)
            #projection(B,i)
            labels=load_data("E:\\twitter_label.txt")
            label_lsh=np.array(labels)
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
                    mean_dict=bay.find_mean(dict_info)
                    std_dict = bay.find_std (dict_info,mean_dict)
                    prior=bay.find_priors(dict_info)

                    #print(len(training_set))
                    #print(len(test_set))

                    _,predictions0=bay.fun_bayes(training_set,test_set,training_label,test_label)
                    acc_funbayes+= _
                    micro_funbayes+=f1_score(test_label,predictions0,average='micro')
                    macro_funbayes+=f1_score(test_label,predictions0,average='macro')

                    print(micro_bayes/5)

                    _,predictions1=nn.fun_knn(training_set,test_set,training_label,test_label)
                    acc_funknn+= _
                    micro_funknn+=f1_score(test_label,predictions1,average='micro')
                    macro_funknn+=f1_score(test_label,predictions1,average='macro')

                    print(micro_bayes/5)

                    _,predictions2=nn.knn(training_set,training_label,test_set,test_label)
                    acc_knn+= _
                    micro_knn+=f1_score(test_label,predictions2,average='micro')
                    macro_knn+=f1_score(test_label,predictions2,average='macro')

                    print(micro_bayes/5)

                    _,predictions3=bay.bayes(mean_dict,std_dict,test_set,test_label,prior)
                    acc_bayes+= _
                    micro_bayes+=f1_score(test_label,predictions3,average='micro')
                    macro_bayes+=f1_score(test_label,predictions3,average='macro')

                    #print(micro_bayes/5)





                file1.write("Test Accuracy on twitter using inbuilt bayes and d =" + str(d) + "::" + str(acc_funbayes/5) +"\n")
                file1.write("Test Accuracy on twitter using inbuilt knn and d =" + str(d) + "::" + str(acc_funknn/5) +"\n")
                file.write("Test Accuracy on twitter using my bayes and d =" + str(d) + "::" + str(acc_bayes/5) +"\n")
                file.write("Test Accuracy on twitter using my knn and d =" + str(d) + "::" + str(acc_knn/5) +"\n \n")

                file1.write("Test Macro F1 Score on twitter using inbuilt bayes and d =" + str(d) + "::" + str(macro_funbayes/5) +"\n")
                file1.write("Test Macro F1 Score on twitter using inbuilt knn and d =" + str(d) + "::" + str(macro_funknn/5) +"\n")
                file.write("Test Macro F1 Score on twitter using my bayes and d =" + str(d) + "::" + str(macro_bayes/5) +"\n")
                file.write("Test Macro F1 Score on twitter using my knn and d =" + str(d) + "::" + str(macro_knn/5) +"\n \n")

                file1.write("Test Micro F1 Score on twitter using inbuilt bayes and d =" + str(d) + "::" + str(micro_funbayes/5) +"\n")
                file1.write("Test Micro F1 Score on twitter using inbuilt knn and d =" + str(d) + "::" + str(micro_funknn/5) +"\n")
                file.write("Test Micro F1 Score on twitter using my bayes and d =" + str(d) + "::" + str(micro_bayes/5) +"\n")
                file.write("Test Micro F1 Score on twitter using my knn and d =" + str(d) + "::" + str(micro_knn/5) +"\n \n")

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
            lsh.start(train_lsh,label_lsh,3)
            #file.close()

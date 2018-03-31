import numpy as np
import pandas as pd 
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix


#exec time: START
start_time=time.time()

############# Data generation and Preprocessing ###############

#reading the Train and Test data from csv file
trainData = pd.read_csv('lab3-train.csv')
testData = pd.read_csv('lab3-test.csv')

#preproccesing Train dataset into features and labels
trainData_x = trainData.iloc[:,:4]
trainData_y = trainData.iloc[:,4]

#preprocessing Test dataset into features and labels
testData_x = testData.iloc[:,:4]
testData_y = testData.iloc[:,4]

#Converting into numpy arrays 
trainData_x = np.array(trainData_x)
trainData_y = np.array(trainData_y)
testData_x = np.array(testData_x)
testData_y = np.array(testData_y)

#print(testData_x)
#print(testData_y)

confusionMatrix=np.zeros((2,2))

############### Model descriptions and Training ################


#Random forest classifier

print("********TASK 1.1: RANDOM FOREST CLASSIFIER******** \n")
RF = RandomForestClassifier(max_depth=4, n_estimators=10,random_state=200)
RF.fit(trainData_x,trainData_y)
predict=RF.predict(testData_x)
score=RF.score(testData_x,testData_y)
S='Overall Accuracy: ' +repr(score*100)+' %'+'\n'
print(S)
confusionMatrix=confusion_matrix(testData_y,predict)
print('Confusion Matrix: ')
print(confusionMatrix)
print('\n')


#AdaBoost Classifier

print("********TASK 1.2: ADA BOOST CLASSIFIER******** \n")
AB=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1,min_samples_leaf=1),n_estimators=11,learning_rate=1.1,random_state=200)
AB.fit(trainData_x,trainData_y)
predict=AB.predict(testData_x)
score=AB.score(testData_x,testData_y)
S='Overall Accuracy: ' +repr(score*100)+' %'+'\n'
print(S)
confusionMatrix=confusion_matrix(testData_y,predict)
print('Confusion Matrix: ')
print(confusionMatrix)
print('\n')


#Individual Classifiers for Voting Classifier
print("********TASK 2.1: INDIVIDUAL CLASSIFIERS for Voting Classifier******** \n")

#Naive Bayes Classifier
print("****Naive Bayes**** \n")
NB=GaussianNB()
NB.fit(trainData_x,trainData_y)
predict=NB.predict(testData_x)
score=NB.score(testData_x,testData_y)
S='Overall Accuracy: ' +repr(score*100)+' %'+'\n'
print(S)
confusionMatrix=confusion_matrix(testData_y,predict)
print('Confusion Matrix: ')
print(confusionMatrix)
print('\n')

#Logistic Regression Classifier
print("****Logistic Regression**** \n")
LR=LogisticRegression(penalty='l2', dual=False, tol=0.00001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=200, solver='liblinear', max_iter=100, multi_class='ovr')
LR.fit(trainData_x,trainData_y)
predict=LR.predict(testData_x)
score=LR.score(testData_x,testData_y)
S='Overall Accuracy: ' +repr(score*100)+' %'+'\n'
print(S)
confusionMatrix=confusion_matrix(testData_y,predict)
print('Confusion Matrix: ')
print(confusionMatrix)
print('\n')

#Decision tree
print("****Decision Tree**** \n")
DT = DecisionTreeClassifier(max_depth=3,min_samples_leaf=1)
DT.fit(trainData_x,trainData_y)
predict=DT.predict(testData_x)
score=DT.score(testData_x,testData_y)
S='Overall Accuracy: ' +repr(score*100)+' %'+'\n'
print(S)
confusionMatrix=confusion_matrix(testData_y,predict)
print('Confusion Matrix: ')
print(confusionMatrix)
print('\n')

#K-Nearest Neighbors Classifier
print("****K-nearest neighbors**** \n")
KNN=KNeighborsClassifier(n_neighbors=2,weights="uniform")
KNN.fit(trainData_x,trainData_y)
predict=KNN.predict(testData_x)
score=KNN.score(testData_x,testData_y)
S='Overall Accuracy: ' +repr(score*100)+' %'+'\n'
print(S)
confusionMatrix=confusion_matrix(testData_y,predict)
print('Confusion Matrix: ')
print(confusionMatrix)
print('\n')

#Neural Network Classifier
print("****Neural Network**** \n")
NN=MLPClassifier(activation='logistic',solver='lbfgs',hidden_layer_sizes=(10,2),momentum = 0.9, learning_rate='constant', learning_rate_init=0.001, early_stopping=True, random_state=200)
NN.fit(trainData_x,trainData_y)
predict=NN.predict(testData_x)
score=NN.score(testData_x,testData_y)
S='Overall Accuracy: ' +repr(score*100)+' %'+'\n'
print(S)
confusionMatrix=confusion_matrix(testData_y,predict)
print('Confusion Matrix: ')
print(confusionMatrix)
print('\n')

#Unweighted Majority Voting Classifier
print("********TASK 2.2: VOTING CLASSIFIER: UNWEIGHTED******** \n")
VCU=VotingClassifier(estimators=[('gnb',NB),('lr',LR),('dt',DT),('knn',KNN),('mlp',NN)], voting = 'soft')
VCU.fit(trainData_x,trainData_y)
predict=VCU.predict(testData_x)
score=VCU.score(testData_x,testData_y)
S='Overall Accuracy: ' +repr(score*100)+' %'+'\n'
print(S)
confusionMatrix=confusion_matrix(testData_y,predict)
print('Confusion Matrix: ')
print(confusionMatrix)
print('\n')

#Weighted Majority Voting Classifier
print("********TASK 2.3: VOTING CLASSIFIER: WEIGHTED******** \n")
p,q,r,s,t=0,0,0,0,0
maxScore=0

#Grid search for weights
for a in range(1,6):
    for b in range(1,6):
        for c in range(1,6):
            for d in range(1,6):
                for e in range(1,6):
                    VCW=VotingClassifier(estimators=[('gnb',NB),('lr',LR),('dt',DT),('knn',KNN),('mlp',NN)], voting = 'soft', weights=[a,b,c,d,e])
                    VCW.fit(trainData_x,trainData_y)
                    score=VCW.score(testData_x,testData_y)
                    if(score>maxScore):
                        maxScore=score
                        p=a
                        q=b
                        r=c
                        s=d
                        t=e


VCW=VotingClassifier(estimators=[('gnb',NB),('lr',LR),('dt',DT),('knn',KNN),('mlp',NN)], voting = 'soft', weights=[p,q,r,s,t])
VCW.fit(trainData_x,trainData_y)
score=VCW.score(testData_x,testData_y)
predict=VCW.predict(testData_x)
S='Weights: '+repr(p)+','+repr(q)+','+repr(r)+','+repr(s)+','+repr(t)+'\n'
print(S)
S='Overall Accuracy: ' +repr(score*100)+' %'+'\n'
print(S)
confusionMatrix=confusion_matrix(testData_y,predict)
print('Confusion Matrix: ')
print(confusionMatrix)
print('\n')


#Unweighted Majority Voting Classifier including Random Forest and AdaBoost
print("********TASK 3.1: VOTING CLASSIFIER: UNWEIGHTED******** \n")
VCU2=VotingClassifier(estimators=[('gnb',NB),('lr',LR),('dt',DT),('knn',KNN),('mlp',NN),('rf',RF),('ab',AB)], voting = 'soft')
VCU2.fit(trainData_x,trainData_y)
predict=VCU2.predict(testData_x)
score=VCU2.score(testData_x,testData_y)
S='Overall Accuracy: ' +repr(score*100)+' %'+'\n'
print(S)
confusionMatrix=confusion_matrix(testData_y,predict)
print('Confusion Matrix: ')
print(confusionMatrix)
print('\n')


#Weighted Majority Voting Classifier including Random Forest and AdaBoost
print("********TASK 3.2: VOTING CLASSIFIER: WEIGHTED******** \n")

p,q,r,s,t,u,v=0,0,0,0,0,0,0
maxScore=0

#Grid search for weights (lower range)
for a in range(1,4):
    for b in range(1,4):
        for c in range(1,4):
            for d in range(1,4):
                for e in range(1,4):
                    for f in range(1,4):
                        for g in range(1,4):
                            VCW2=VotingClassifier(estimators=[('gnb',NB),('lr',LR),('dt',DT),('knn',KNN),('mlp',NN),('rf',RF),('ab',AB)], voting = 'soft', weights=[a,b,c,d,e,f,g])
                            VCW2.fit(trainData_x,trainData_y)
                            score=VCW2.score(testData_x,testData_y)
                            if(score>maxScore):
                                maxScore=score
                                p=a
                                q=b
                                r=c
                                s=d
                                t=e
                                u=f
                                v=g
VCW2=VotingClassifier(estimators=[('gnb',NB),('lr',LR),('dt',DT),('knn',KNN),('mlp',NN),('rf',RF),('ab',AB)], voting = 'soft', weights=[p,q,r,s,t,u,v])
VCW2.fit(trainData_x,trainData_y)
score=VCW2.score(testData_x,testData_y)
predict=VCW2.predict(testData_x)
S='Weights: '+repr(p)+','+repr(q)+','+repr(r)+','+repr(s)+','+repr(t)+','+repr(u)+','+repr(v)+'\n'
print(S)
S='Overall Accuracy: ' +repr(score*100)+' %'+'\n'
print(S)
confusionMatrix=confusion_matrix(testData_y,predict)
print('Confusion Matrix: ')
print(confusionMatrix)
print('\n')


#exec time: END
print("--- Total training time %s (s) ---" % (time.time() - start_time))

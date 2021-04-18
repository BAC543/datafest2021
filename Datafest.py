#Datafest 2021
#KNN (k Nearest Neighbors) Analysis
#Are health care workers under x conditions likely to abuse x drug?

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from matplotlib.pyplot import figure

from sklearn import tree
import matplotlib.pyplot as plt
model=GaussianNB()

dataset = pd.read_csv(r'C:\Users\arjun\Documents\us19.csv')
# split dataset
#dataprep
##############################Inputs###############################################|
#iloc[:, col-x to col-y]                                                           |
#Health Care Workers & Mental Health Disorders                                     |
#Columns LF - LR                                                                   |
x = dataset.iloc[:,[346,317,318,319,320,321,322,323,324,325,326,327,328,329,330]] #|
#print(x)                                                                          #|
#                                                                                 #|
##############################OUTPUT#######################################|#######|
#Column KF  = DAST1                                                                |
#DAST 1- Have you used drugs other than those required for medical reasons?        |
#1 = yes  0 = no                                                                   |
y = dataset.iloc[:, 291]                                                          #|
###################################################################################|

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = .2)
# Feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#x_train - holds all instances with atributes
#y_train - holds all instances with labels

#Innit KNN
classifier = KNeighborsClassifier(n_neighbors = 11, p = 2, metric = 'euclidean').fit(x_train,y_train)

#Predict the test set results
#
y_pred = classifier.predict(x_test)
#print('----------------Prediction------------')
#print(y_pred)

#tests data to find false postive and false
#print('---------------Test------------')
#print(y_test)

############################Confusion Matrix############################
#
#------Actual Values
#P   V|     |      |
#r   a| TP  |  FP  |
#e   l|_____|______|
#d   u|     |      |
#i   e| FN  |  TN  |
#c   s|_____|______|
#t
#e
#d
print('-----------------CONFUSTION MATRIX-----------------')
cm = confusion_matrix(y_test, y_pred)
print(cm)

###########################F1 Score################################################
print('-----------------F1 SCORE-----------------')
print(f1_score(y_test, y_pred, pos_label='positive' ,average = 'weighted'))

###########################Accuracy Score###########################################
print('-----------------ACCURACY SCORE-----------------')
print(accuracy_score(y_test, y_pred))
#Naive Bayes 
print(model.fit(x_train,y_train))
print(model.score(x_test,y_test))
clf = tree.DecisionTreeClassifier()
#Decesion Tree
#320 is bipolar and 322 is depresion
x = dataset.iloc[:,[320,322,346]]
y = dataset.iloc[:, 291]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = .2)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
clf = clf.fit(x_test, y_test)
figure(figsize=(20, 20), dpi=40)
tree.plot_tree(clf)

plt.show()

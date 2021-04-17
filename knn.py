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


dataset = pd.read_csv(r'C:\Users\Brian Coppola\datathon-workspace\US\US\us19.csv')
print(len(dataset))
print(dataset.head())

# split dataset

#dataprep

#inputs
#iloc[:, col-x to col-y]
#Answers for drug abuse
#Columns KF to KO
x = dataset.iloc[:, 291:301]
print(x)
#output
y = dataset.iloc[:, 348: 354]
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = .2)
#print(train_test_split(x, y, random_state = 0, test_size = .2))
# Feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

#x_train - holds all instances with atributes
#y_train - holds all instances with labels

#Innit KNN
classifier = KNeighborsClassifier(n_neighbors = 11, p = 2, metric = 'euclidean').fit(x_train,y_train)

#Predict the test set results
y_pred = classifier.predict(x_test)
print(y_pred)

#confusion matrix
#out
#------predicted
#actual|
#

print('-----------------CONFUSTION MATRIX-----------------')
cm = confusion_matrix(y_test, y_pred)
print(cm)

#f1 score
print('-----------------F1 SCORE-----------------')
print(f1_score(y_test, y_pred, pos_label='positive' ,average = 'micro'))

#accuracy score
print('-----------------ACCURACY SCORE-----------------')
print(accuracy_score(y_test, y_pred))






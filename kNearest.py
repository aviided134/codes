import pandas as pd
import numpy as np 
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split

df = pd.read_csv('breast-cancer-wisconsin. data.txt')
df.replace('?', -999999, inplace = True)  # -99999 is considered as an outlier. So most of the algorithm will ignore it
df.drop(['id'], 1, inplace = True)  # 1 is the axis. '0' for row and '1' for column


X = np.array(df.drop(['class'], 1))  # x for the feature and y for the label or class
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)


example_measures = np.array([[1,1,1,1,7,8,7,1,1], [1,1,1,1,1,1,2,1,1]])

#example_measures = np.array([[10,1,1,1,7,10,1,1,1], [4,1,1,1,9,7,8,10,1]])
example_measures = example_measures.reshape(len(example_measures),-1)  # converting in 1D array from 2-D array. 

prediction=clf.predict(example_measures)
print(prediction)

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


data_dir = os.path.join('.','data')
model_dir = os.path.join('.','models')


knn = KNeighborsClassifier(algorithm='auto', metric='euclidean', n_neighbors=10, weights='distance')
lr = LogisticRegression(C=0.0001, penalty='l1', solver='liblinear')
rf = RandomForestClassifier(n_estimators=20, min_samples_split=3, min_samples_leaf=3, max_leaf_nodes=5, max_features='log2', max_depth=None, criterion='entropy')
svc = SVC(kernel='rbf', gamma='scale', C=0.001)
mlp = MLPClassifier(solver='adam', learning_rate='adaptive', hidden_layer_sizes=50, alpha=0.1, activation='relu')

df = pd.read_csv(os.path.join(data_dir,'feature_set.csv'))

X = df.drop('Class', axis=1)
y = df['Class']

knn.fit(X,y)
knn_name = os.path.join(model_dir,'knn.sav')
pickle.dump(knn, open(knn_name, 'wb'))
# print('KNN saved!')

lr.fit(X,y)
lr_name = os.path.join(model_dir,'logistic regression.sav')
pickle.dump(lr, open(lr_name, 'wb'))
# print('Logistic Regression saved!')

rf.fit(X,y)
rf_name = os.path.join(model_dir,'random forest.sav')
pickle.dump(rf, open(rf_name, 'wb'))
# print('Random Forest saved!')

svc.fit(X,y)
svc_name = os.path.join(model_dir,'svc.sav')
pickle.dump(svc, open(svc_name, 'wb'))
# print('SVC saved!')

mlp.fit(X,y)
mlp_name = os.path.join(model_dir,'mlp.sav')
pickle.dump(mlp, open(mlp_name, 'wb'))
# print('MLP saved!')
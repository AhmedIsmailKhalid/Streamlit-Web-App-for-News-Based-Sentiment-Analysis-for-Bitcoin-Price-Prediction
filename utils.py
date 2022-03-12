import pandas as pd
import numpy as np
import nltk
import itertools
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle
import warnings
warnings.filterwarnings('ignore')
import requests
import regex as re
from bs4 import BeautifulSoup



# Machine Learning Library Imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor 
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score,f1_score, confusion_matrix, plot_confusion_matrix,ConfusionMatrixDisplay, r2_score, mean_squared_error, mean_absolute_error,mean_absolute_percentage_error



def remove_punctuation(text):
    nltk.download("punkt") # Commenting this part since it is already installed once
    words = nltk.word_tokenize(text)
    text = [word for word in words if word.isalnum()]
    text = ' '.join(text)
    return text


def lemmatize_text(text):
    # First create/instantiate the lemmatizer
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return ' '.join(lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text))

def batch_generator(iterable, batch_size=1000):
    iterable = iter(iterable)

    while True:
        batch = list(itertools.islice(iterable, batch_size))
        if len(batch) > 0:
            yield batch
        else:
            break


def clf_highlight(s):
    if s['Train Accuracy'] - s['Test Accuracy'] >= 0.2:
        return ['color:red;']*s.shape[0] 
    elif s['Train Accuracy'] < 0.7:
        return ['color:blue;']*s.shape[0]
    elif s['Train Accuracy'] < s['Test Accuracy'] :
        return ['color:brown;']*s.shape[0]
    else:
        return ['color: black']*s.shape[0]
    


def no_cv (feature_set, scale, fs, DFS) :
    data = []
    # Create a list of classifiers with their default parameters. This list will be used to fit the models using for loop
    classifiers = [BernoulliNB(), KNeighborsClassifier(), LogisticRegression(), RandomForestClassifier(), SVC(), MLPClassifier()]
       
    # Create a list of names of classifiers created above. This will be used to print their respective accuracies
    classifier_names = ['Bernoulli NB', 'KNN', 'Logistic Regression', 'Random Forest', 'SVC', 'MLP']
    
    # Get X (features) and y (label) from the feature_set accepted as argument
    X = feature_set.drop('Class',axis=1)
    y = feature_set['Class']
    
    # Create train and test(validation) sets for both X and y with 75%/25% splits 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)
    
    if scale=='standard' or scale=='Standard':
        scalar = StandardScaler()
        X_train = scalar.fit_transform(X_train)
        X_test = scalar.transform(X_test)
        
    if scale=='minmax' or scale=='MinMax':
        scalar = MinMaxScaler()
        X_train = scalar.fit_transform(X_train)
        X_test = scalar.transform(X_test)
        
    if scale=='robust' or scale=='Robust':
        scalar = RobustScaler()
        X_train = scalar.fit_transform(X_train)
        X_test = scalar.transform(X_test)

    # Use a for loop to iterate over the classifiers and classifiers_names lists to create and fit the models
    for classifier, classifier_name in zip(classifiers, classifier_names) :
        # Create the model
        clf = classifier

        # Fit the model
        clf.fit(X_train, y_train)

        # Get predictions
        y_pred = clf.predict(X_test)

        # Calculate the accuracy and recall of the model
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Print the Name of the classifier and the result of each of the evaluation metrics
        print('\t',classifier_name)
        print('Train Accuracy :', train_acc)
        print('Test Accuracy :', acc)
        print('Recall :', recall)
        print('Precision :', precision)
        print('F1 :', f1)
        #print('Actual      :', np.array(y_test))
        #print('Predictions :', y_pred)
        print('Confusion Matrix :\n')
        fig, ax = plt.subplots(figsize=(6,6))
        plot_confusion_matrix(clf, X_test, y_test, cmap='Purples', values_format='d', ax=ax);
        plt.show()
        print()
        
        data.append([scale, fs, classifier_name, train_acc, acc, recall, precision, f1])
    
    if scale=='' or scale=='no scale':
        print('Table of models and results of their evaluation metrics while using No Scaling')
    elif scale=='minmax' or scale=='MinMax':
        print('Table of Models and results of their Evaluation metrics while using MinMaxScaler')
    else:
         print('Table of Models and results of their Evaluation metrics while using', scale.capitalize() + 'Scaler')
    
    display(pd.DataFrame(data,columns=['Scale', 'Feature Set', 'Name', 'Train Accuracy', 'Test Accuracy', 'Recall', 'Precision', 'F1']).drop(['Feature Set','Scale'], axis=1).style.apply(clf_highlight, axis=1))
    DFS.append(pd.DataFrame(data,columns=['Scale', 'Feature Set', 'Name', 'Train Accuracy', 'Test Accuracy', 'Recall', 'Precision', 'F1']))
    
    
    
def no_cv_params (feature_set, DFS, clfs) :
    data = []
    # Create a list of classifiers with their default parameters. This list will be used to fit the models using for loop
    classifiers = clfs
   
    # Create a list of names of classifiers created above. This will be used to print their respective accuracies
    classifier_names = ['Bernoulli NB', 'KNN', 'Logistic Regression', 'Random Forest', 'SVC', 'MLP']
    
    # Get X (features) and y (label) from the feature_set accepted as argument
    X = feature_set.drop('Class',axis=1)
    y = feature_set['Class']
    
    # Create train and test(validation) sets for both X and y with 75%/25% splits 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=8)
    
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
        
    
    # Use a for loop to iterate over the classifiers and classifiers_names lists to create and fit the models
    for classifier, classifier_name in zip(classifiers, classifier_names) :
        # Create the model
        clf = classifier

        # Fit the model
        clf.fit(X_train, y_train)

        # Get predictions
        y_pred = clf.predict(X_test)

        # Calculate the accuracy and recall of the model
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Print the Name of the classifier and the result of each of the evaluation metrics
        print('\t',classifier_name)
        print('Train Accuracy :', train_acc)
        print('Test Accuracy :', acc)
        print('Recall :', recall)
        print('Precision :', precision)
        print('F1 :', f1)
        #print('Actual      :', np.array(y_test))
        #print('Predictions :', y_pred)
        print('Confusion Matrix :\n')
        fig, ax = plt.subplots(figsize=(6,6))
        plot_confusion_matrix(clf, X_test, y_test, cmap='Reds', values_format='d', ax=ax);
        plt.show()
        print()
        
        data.append([classifier_name, train_acc, acc, recall, precision, f1])
    
    display(pd.DataFrame(data,columns=['Name', 'Train Accuracy', 'Test Accuracy', 'Recall', 'Precision', 'F1']).style.apply(clf_highlight, axis=1))
    DFS.append(pd.DataFrame(data,columns=['Name', 'Train Accuracy', 'Test Accuracy', 'Recall', 'Precision', 'F1']))
    
    
def clf_cv (feature_set, DFS) :
    data = []
    
    # Get X (features) and y (label) from the feature_set accepted as argument
    X = feature_set.drop('Class',axis=1)
    y = feature_set['Class']
    
    tscv = TimeSeriesSplit(gap=0, n_splits=113)
 
    train_accs = []
    test_accs = []
    recalls = []
    precisions = []
    f1s = []
    
    
    classifiers = [BernoulliNB(), KNeighborsClassifier(), LogisticRegression(), RandomForestClassifier(), SVC(), MLPClassifier()] 

    classifier_names = ['Bernoulli NB', 'KNN', 'Logistic Regression', 'Random Forest', 'SVC', 'MLP']

    for classifier, classifier_name in zip(classifiers, classifier_names):
        for train_index, test_index in tscv.split(X, y):
            X_train, X_test = X.values[train_index], X.values[test_index]
            y_train, y_test = y.values[train_index], y.values[test_index]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            clf = classifier
            clf.fit(X_train, y_train)

            y_pred_train = clf.predict(X_train)
            y_pred = clf.predict(X_test)

            train_accs.append(accuracy_score(y_train, y_pred_train))
            test_accs.append(accuracy_score(y_test, y_pred))
            recalls.append(recall_score(y_test, y_pred))
            precisions.append(precision_score(y_test, y_pred, labels=np.unique(y_pred)))
            f1s.append(f1_score(y_test, y_pred))
            
        
        data.append([classifier_name, np.mean(np.array(train_accs).ravel()), np.mean(np.array(test_accs).ravel()),\
                        np.mean(np.array(recalls).ravel()), np.mean(np.array(precisions).ravel()),\
                        np.mean(np.array(f1s).ravel())])
        

#         print('Train Accuracy   :', np.mean(np.array(train_accs).ravel()))
#         print('Test Accuracy    :', np.mean(np.array(test_accs).ravel()))
#         print('Recall Score     :', np.mean(np.array(recalls).ravel()))
#         print('Precision Score  :', np.mean(np.array(precisions).ravel()))
#         print('F1 Score         :', np.mean(np.array(f1s).ravel()))
#         print('\n')

    #display(pd.DataFrame(data, columns=['Name', 'Train Accuracy', 'Test Accuracy', 'Recall', 'Precision', 'F1']))
    DFS.append((pd.DataFrame(data, columns=['Name', 'Train Accuracy', 'Test Accuracy', 'Recall', 'Precision', 'F1'])))
    
    
    
def clf_cv_params (X_train, y_train, X_test, y_test, clfs) :
    data = []
    
    # Get X (features) and y (label) from the feature_set accepted as argument
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    test_accs = []
    recalls = []
    precisions = []
    f1s = []
    #tns = []
    #fps = []
    #fns = []
    #tps = []
    
    classifiers = clfs 

    classifier_names = ['KNN', 'Logistic Regression', 'Random Forest', 'SVC', 'MLP']

    for classifier, classifier_name in zip(classifiers, classifier_names):
        clf = classifier
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        conf_mat = confusion_matrix(y_test, y_pred)
        
        test_accs.append(accuracy_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        #tns.append(conf_mat[0][0])
        #fps.append(conf_mat[0][1])
        #fns.append(conf_mat[1][0])
        #tps.append(conf_mat[1][1])

        print('\t\t',classifier_name)
        fig, ax = plt.subplots(figsize=(6,6))
        plot_confusion_matrix(clf, X_test, y_test, cmap='Purples', values_format='d', ax=ax);
        plt.show()

        
        data.append([classifier_name, np.mean(np.array(test_accs).ravel()),\
                        np.mean(np.array(recalls).ravel()), np.mean(np.array(precisions).ravel()),\
                        np.mean(np.array(f1s).ravel())])#, tps, fps, tns, fns])
        

#         print('Train Accuracy   :', np.mean(np.array(train_accs).ravel()))
#         print('Test Accuracy    :', np.mean(np.array(test_accs).ravel()))
#         print('Recall Score     :', np.mean(np.array(recalls).ravel()))
#         print('Precision Score  :', np.mean(np.array(precisions).ravel()))
#         print('F1 Score         :', np.mean(np.array(f1s).ravel()))
#         print('\n')

    #display(pd.DataFrame(data, columns=['Name', 'Accuracy', 'Recall', 'Precision', 'F1']))
    DFS = pd.DataFrame(data, columns=['Name', 'Accuracy', 'Recall', 'Precision', 'F1'])
    return DFS

import os
import pickle
from pyexpat import model
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

def predict() :
    model_dir = os.path.join('.','models')
    
    
    uploaded_path = os.path.join('.','uploaded data')
    data_path = os.path.join('.','data')
    
    uploaded_files_dir = [f for f in os.listdir(uploaded_path) if os.path.isfile(os.path.join(uploaded_path, f))] 
    saved_models = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]

    col1, col2 = st.columns(2)

    with col1 :
       model = st.selectbox('Select your model to serve', ['']+saved_models)

    with col2 :
       file = st.selectbox('Select file for predicting', ['']+uploaded_files_dir)

    clf = pickle.load(open(os.path.join('.','models', model), 'rb'))
    # clf = pickle.load(open(os.path.join('.','models', filename+'.sav'), 'wb'))
    

    df = pd.read_csv(os.path.join('uploaded data',file))
    
    X = df.drop('Class', axis=1) 
    y = df['Class']

    y_pred = clf.predict(X)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    st.write('Accuracy :', acc, 'F1 Score :', f1)
    st.write(y_pred)
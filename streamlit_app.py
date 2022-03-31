#when we import hydralit, we automatically get all of Streamlit
import os
import sys
import pickle
import subprocess
import glob
import nltk
import streamlit as st
import hydralit as hy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from io import StringIO
from nltk.corpus import stopwords, wordnet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from home_page import homepage
from data import get_data
from train_models import train
from eda import perform_eda
from feature_engineering import create
from hyperparameter_tuning import tune
from process_uploaded_data import process
from serve import predict


from collections import Counter
from wordcloud import WordCloud
from st_aggrid import AgGrid

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

app = hy.HydraApp(title='News Based Sentiment analysis for Bitcoin Price Prediction', navbar_theme={'txc_inactive': '#FFFFFF','menu_background':'orange','txc_active':'#FFFFFF'})

nltk.download('stopwords')
nltk.download('wordnet')

# hide_menu_style = """
#         <style>
#         #MainMenu {visibility: hidden;}
#         </style>
#         """
# st.markdown(hide_menu_style, unsafe_allow_html=True)

st.set_option('deprecation.showPyplotGlobalUse', False)

print('\n\nLibs and utils imported successfully!')

data_dir = os.path.join('.','data')
model_dir = os.path.join('.','models')

news_df = pd.read_csv(os.path.join(data_dir,'NEWS DF.csv'))
btc_df = pd.read_csv(os.path.join(data_dir,'Gemini_BTCUSD_1hr.csv'), skiprows=1)

@app.addapp(is_home=True)
def home():
    homepage()
    
@app.addapp(title='Data')
def data():
    get_data(news_df, btc_df)


@st.cache(persist=True, suppress_st_warning=True)
@app.addapp(title='Exploratory Data Analysis')
def eda():
    perform_eda()


@st.cache(persist=True, suppress_st_warning=True)
@app.addapp(title='Train Models')
def train_model():
    # df = pd.read_csv(os.path.join(data_dir,'feature_set.csv'))
    df = pd.read_csv(os.path.join(data_dir,'FEATURE SET.csv'))
    
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
    st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)

    # train_options = st.radio('Training Options', ('Use default feature set', 'Upload new feature set', 'Create new feature set', 'Perform Hyperparameter Tuning'))
    train_options = st.radio('Training Options', ('Use default feature set', 'Use created feature set(s)', 'Perform Hyperparameter Tuning'))

    if train_options == 'Use default feature set' :
        train(option='Default')
    
    if train_options == 'Perform Hyperparameter Tuning' :
        tune(df)

    if train_options == 'Use created feature set(s)' :
        try :
            train('Custom')
        except :
            st.write('')
        

@st.cache(persist=True, suppress_st_warning=True)
@app.addapp(title='Serve Predictions')
def evaluate():
    try :
        predict()
    except :
        st.write('')
    

#Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.
app.run()
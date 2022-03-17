#when we import hydralit, we automatically get all of Streamlit
from email import header
import os
import sys
import pickle
import subprocess
import streamlit as st
import hydralit as hy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from io import StringIO
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from load_data import load
from train_models import train
from eda import perform_eda
from feature_engineering import create
from hyperparameter_tuning import tune
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

app = hy.HydraApp(title='News Based Sentiment analysis for Bitcoin Price Prediction', navbar_theme=None)

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

news_df, btc_df = load()

@app.addapp(is_home=True)
def my_home():
    hy.write('THIS IS THE HOME PAGE! ADD THE BITCOIN REALTIME PRICE AS WELL AS CHARTS IF POSSIBLE. ELSE JUST ADD SOME INFO ABOUT THE WEBSITE')
    
@app.addapp(title='Data')
def data():
    # hy.markdown('The raw news dataset')
    # hy.dataframe(news_df.drop('temp_list', axis=1))
    # hy.markdown('\n\nThe bitcoin price datarset')
    # st.dataframe(btc_df)

    col1, col2 = st.columns(2)

    with col1 :
        st.title('Raw news dataset')
        st.dataframe(news_df.drop('temp_list', axis=1))

    with col2 :
        st.title('Raw bitcoin dataset')
        st.dataframe(btc_df)


@st.cache(persist=True, suppress_st_warning=True)
@app.addapp(title='Exploratory Data Analysis')
def eda():
    perform_eda(news_df)


@st.cache(persist=True, suppress_st_warning=True)
@app.addapp(title='Train Models')
def train_model():
    df = pd.read_csv(os.path.join(data_dir,'feature_set.csv'))
    
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
    st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)

    # train_options = st.radio('Training Options', ('Use default feature set', 'Upload new feature set', 'Create new feature set', 'Perform Hyperparameter Tuning'))
    train_options = st.radio('Training Options', ('Use default feature set', 'Create new feature set', 'Perform Hyperparameter Tuning'))

    if train_options == 'Use default feature set' :
        train()
    
    if train_options == 'Perform Hyperparameter Tuning' :
        tune(df)

    if train_options == 'Create new feature set' :
        try :
            created_news_df, created_btc_df, feature_set = create(df)
            col1, col2, col3 = st.columns(3)
            with col1 :
                st.dataframe(created_news_df)
            with col2 :
                st.dataframe(created_btc_df)
            with col3 :
                st.dataframe(feature_set)
        except TypeError :
            st.write('')
        

@st.cache(persist=True, suppress_st_warning=True)
@app.addapp(title='Evaluate Performance')
def evaluate():
    st.write('Evaluation Coming Soon!')
    

#Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.
app.run()
# Import all other libraries stored in libs.py. This is done to make this app.py file cleaner and easier to read and understand

# from libspackages import *

# Helper Functions. This py script also has all the imports for Machine Learning Algoritms as well as
# other neccessary imports. This script has functions that we defined that will be used in our project
from utils import *
from load_data import load_data
from processing import preprocess
from eda import *
from st_aggrid import AgGrid
import streamlit as st

import os
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


st.set_option('deprecation.showPyplotGlobalUse', False)

print('\n\nLibs and utils imported successfully!')

data_dir = os.path.join('.','data')
model_dir = os.path.join('.','models')

# unless function or arguments change, cache outputs to disk and use cached outputs anytime app re-runs
def main() :
    st.title('News Based Sentiment Analysis System for Bitcoin')
    st.markdown('Testing Elly Kang Streamlit Website')

    st.write('Libs and utils imported successfully!')

    news_df, btc_df = load_data()
    # news_df, btc_df = process_data(news_df, btc_df)

    feature_set = pd.read_csv(os.path.join(data_dir,'feature_set.csv'))

    
    
    st.dataframe(feature_set.head())


@st.cache(persist=True, suppress_st_warning=True)
def process_data(news_df, btc_df) :
    print('loading feature set')
    
    news_df, btc_df = preprocess(news_df, btc_df)

    return news_df, btc_df

@st.cache(persist=True, suppress_st_warning=True)
def load_model():
    mlp_name = os.path.join(model_dir,'mlp.sav')
    mlp = pickle.load(open(mlp_name, 'rb'))

    return mlp


def eda_visualize(news_df, btc_df) :
    top15, wordcloud = visualize(news_df, btc_df)

    st.dataframe(top15)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title('\nWordcloud for news articles related to Bitcoin\n')
    plt.show()
    st.pyplot()

    return top15


if __name__ == '__main__' :
    main()
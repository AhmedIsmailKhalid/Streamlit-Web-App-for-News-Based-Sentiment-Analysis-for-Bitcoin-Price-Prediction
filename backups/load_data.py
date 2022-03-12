import os
import pandas as pd
import streamlit as st
from utils import remove_punctuation, lemmatize_text
from nltk.corpus import stopwords

data_dir = os.path.join('.','data')


@st.cache(persist=True, suppress_st_warning=True, allow_output_mutation=True)
def load() :
     news_df = pd.read_csv(os.path.join(data_dir,'news_df.csv'))
     btc_df = pd.read_csv(os.path.join(data_dir,'prices/gemini_BTCUSD_1hr.csv'))

     stop_words = stopwords.words('english')
     stop_words.append('https')
     stop_words.append('nbsp')

     news_df['Date'] = pd.to_datetime(news_df['Date'], utc=True)
     news_df['Time'] = news_df['Date'].dt.round('H').dt.time
     news_df['Date'] = news_df['Date'].dt.date
     news_df = news_df[['Date', 'Time', 'Text']]
     news_df = news_df.sort_values('Date', ascending=False).reset_index(drop=True)
     
     to_replace = 'Sign up for our newslettersBy signing up, you will receive emails about CoinDesk products and you agree to our terms &'\
' conditions and privacy policySign up for our newslettersBy signing up, you will receive emails about CoinDesk products and you'\
' agree to our terms & conditions and privacy policyPlease consider using a different web browser for better experience.'

     news_df = news_df.replace(regex=True,to_replace=to_replace,value=r'')


     news_df['Date'] = news_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
     news_df['Time'] = news_df['Time'].apply(lambda x: x.strftime('%H-%M-%S'))

     btc_df['Date'] = pd.to_datetime(btc_df['Date'], utc=True)
     btc_df['Time'] = btc_df['Date'].dt.time
     btc_df['Date'] = btc_df['Date'].dt.date
     btc_df = btc_df[['Unix Timestamp', 'Date', 'Time', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
     btc_df = btc_df.sort_values('Date', ascending=False).reset_index(drop=True)

     btc_df['Date'] = btc_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
     btc_df['Time'] = btc_df['Time'].apply(lambda x: x.strftime('%H-%M-%S'))

     news_df['Text'] = news_df['Text'].str.lower()
     news_df['Text'] = news_df['Text'].apply(remove_punctuation)

     stop_words = stopwords.words('english')
     stop_words.append('https')
     stop_words.append('nbsp')

     news_df['Text'] = news_df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

     news_df['Text'] = news_df['Text'].apply(lemmatize_text)

     news_df = news_df[news_df.Text.str.contains("bitcoin")].reset_index(drop=True)

     news_df['temp_list'] = news_df['Text'].apply(lambda x:str(x).split())

     return news_df, btc_df

import os
import pandas as pd
import regex
from nltk.corpus import stopwords

data_dir = os.path.join('.','data')


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
print(to_replace)

news_df = news_df.replace(regex=True,to_replace=to_replace,value=r'')

print(news_df.head())
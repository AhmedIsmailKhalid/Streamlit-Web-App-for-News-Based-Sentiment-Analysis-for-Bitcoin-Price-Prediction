import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud, ImageColorGenerator
from nltk.corpus import stopwords

def visualize(news_df, btc_df) :
    stop_words = stopwords.words('english')
    print(stop_words)
    stop_words.append('https')
    stop_words.append('nbsp')

    news_df['Text'] = news_df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    news_df = news_df[news_df.Text.str.contains("bitcoin")].reset_index(drop=True)

    news_df['temp_list'] = news_df['Text'].apply(lambda x:str(x).split())
    top = Counter([item for sublist in news_df['temp_list'] for item in sublist])
    temp = pd.DataFrame(top.most_common(15))
    temp.columns = ['Word','Frequency']
    top15 = temp.style.background_gradient(cmap='Purples')


    wordcloud = WordCloud(background_color='white', colormap='inferno').generate(' '.join(news_df['Text']))
    
    return top15, wordcloud

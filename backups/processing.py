import pandas as pd
import numpy as np
import regex as re
from utils import *
from utils import remove_punctuation, batch_generator, lemmatize_text
from nltk.corpus import stopwords, wordnet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def preprocess(news_df, btc_df) :
    Analyzer = SentimentIntensityAnalyzer()

    news_df['Text'] = news_df['Text'].str.lower()
    news_df['Text'] = news_df['Text'].apply(remove_punctuation)

    stop_words = stopwords.words('english')
    stop_words.append('https')
    stop_words.append('nbsp')

    news_df['Text'] = news_df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    news_df['Text'] = news_df['Text'].apply(lemmatize_text)

    news_df = news_df[news_df.Text.str.contains("bitcoin")].reset_index(drop=True)

    news_df['temp_list'] = news_df['Text'].apply(lambda x:str(x).split())

    ''' STATISTICAL DRIVIEN APPROACH'''

    words = []
    for row in news_df['temp_list']:
        words = words + row

    words = np.unique(np.array(words))

    words = [re.split('(\d+)', word) for word in words]

    stop_words.append('k')
    stop_words.append('s')
    stop_words.append('x')
    stop_words.append('m')
    stop_words.append('')

    words_ = []
    for sublist in words:
        for word in sublist :
            if word not in stop_words:
                words_.append(word)

    words = words_
    del(words_) 

    words = [word for word in words if len(word) > 3]

    btc_df = btc_df[btc_df['Date'].isin(news_df['Date'])]
    btc_df = btc_df.sort_values('Date',ascending=False).reset_index(drop=True)
    
    filtered_news_counts = news_df.groupby(['Date','Time']).count()
    filtered_news_counts = filtered_news_counts.sort_values('Time',ascending=False).reset_index()

    lst = []
    for row,count in zip(btc_df.iterrows(),filtered_news_counts['Text']) :
        for i in range(count) :
            lst.append(row[1])
            

    btc_df2 = pd.DataFrame(lst)
    btc_df2 = btc_df2.reset_index(drop=True)

    scores = []
    for value in btc_df2[['Close','Open']].iterrows() :
        score = (value[1][0] - value[1][1]) / btc_df2['Close'].max()
        scores.append(score)

    news_df['Fluctuation Scores'] = scores
    news_df = news_df.fillna(0)

    batches = []

    for batch in batch_generator(words, 10):
        batches.append(batch)

    
    scores_dict = {}
    WORDS = 0
    for word in words :
        word_scores = []
        #print('\t\t\t\t',word, '\n')
        #print(word)
        WORDS += 1 
        for batch in batches:
            #print(Text)
            if word in batch :
                index = news_df.index[news_df['Text'].str.contains(word)]
                date = news_df.iloc[index]['Time']
                score = news_df.iloc[index]['Fluctuation Scores']
                word_scores.append(score)
            #if word not in Text :
            #    print('FOUND : No')
        word_score = np.mean(word_scores)
        #print(word, word_score)
        scores_dict[word] = word_score
        #print('***********************************************')
        if WORDS%1000 == 0 :
            print(str(WORDS) + '/' + str(len(words)) + ' words processed')


    scores_dict_df = pd.DataFrame(data = {'Word' : list(scores_dict.keys()), 'Score' : list(scores_dict.values())})

    scaled_scores = []
    for s in scores_dict_df['Score'] :
        scaled_score = (s - (-4)) / (4 - (-4))
        scaled_scores.append(scaled_score)

    scores_dict_df['Scaled Scores'] = scaled_scores

    analyzer = SentimentIntensityAnalyzer()
    for word in scores_dict_df['Word'] :
        try :    
            analyzer.lexicon.pop(word)
        except KeyError :
            continue

    new_words = {}

    for row in scores_dict_df.iterrows() :
        new_words[row[1][0]] = row[1][2]

    analyzer.lexicon.update(new_words)

    btc_df2 = btc_df2[['Date','Close']]
    btc_df2 = btc_df2.sort_values('Date',ascending=False)
    btc_df2['Price Difference'] = btc_df2['Close'].diff().fillna(0)

    labels = []

    for v in btc_df2['Price Difference'] :
        if v<0 :
            labels.append(-1)
        elif v>0 :
            labels.append(1)
        else :
            labels.append(-99)
        

    btc_df2['Class'] = labels
    btc_df2 = btc_df2.reset_index(drop=True)#.sort_values('Date',ascending=False)
    btc_df2 = btc_df2.drop('Price Difference', axis=1)  

    btc_df2['1 Hr Diff'] = btc_df2['Close'].diff()
    btc_df2['2 Hr Diff'] = btc_df2['Close'].diff(2)
    btc_df2['3 Hr Diff'] = btc_df2['Close'].diff(3)
    btc_df2['4 Hr Diff'] = btc_df2['Close'].diff(4)
    btc_df2['5 Hr Diff'] = btc_df2['Close'].diff(5)
    btc_df2['6 Hr Diff'] = btc_df2['Close'].diff(6)
    btc_df2['7 Hr Diff'] = btc_df2['Close'].diff(7)
    btc_df2['7 Hr MA'] = btc_df2['Close'].rolling(7).mean()  
    
    feature_set = pd.concat([btc_df2.drop(['Date','Close'],axis=1),
                         news_df.drop(['Date', 'Time', 'Text', 'temp_list','Fluctuation Scores'], axis=1)], axis=1)

    # Drop the rows with NaN values
    feature_set = feature_set.dropna()
    feature_set = feature_set[feature_set['Class']!=-99].reset_index(drop=True)

    return news_df, btc_df
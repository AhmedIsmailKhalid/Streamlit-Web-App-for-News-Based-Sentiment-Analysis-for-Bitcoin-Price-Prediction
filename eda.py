import os
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from stqdm import stqdm
from st_aggrid import AgGrid
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils import remove_punctuation, lemmatize_text


def perform_eda() :  
    uploaded_path = os.path.join('.','uploaded data')
    files_dir = [f for f in os.listdir(uploaded_path) if os.path.isfile(os.path.join(uploaded_path, f))]

    file = st.selectbox('Select File to Display', ['']+files_dir)

    news_df =  pd.read_csv(os.path.join('uploaded data',file))

    placeholder = st.empty()

    with placeholder.container() :
        st.info('Processing Data. This may take a while depending on the size of the file. Please be patient. Thank you 🙂')
    
    '''Top 15 Most Frequent Words'''
    Analyzer = SentimentIntensityAnalyzer()

    ########################################################################################################################
    news_df['Text'] = news_df['Text'].str.lower()
    news_df['Text'] = news_df['Text'].apply(remove_punctuation)

    stop_words = stopwords.words('english')

    stop_words.append('https')
    stop_words.append('nbsp')

    #news = news_df.copy(deep=True)

    news_df['temp_list'] = news_df['Text'].apply(lambda x:str(x).split())

    news_df['Text'] = news_df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    news_df['Text'] = news_df['Text'].apply(lemmatize_text)

    news_df[news_df.Text.str.contains("bitcoin")].reset_index(drop=True, inplace=True)

    news_df['temp_list'] = news_df['Text'].apply(lambda x:str(x).split())
    top15 = Counter([item for sublist in news_df['temp_list'] for item in sublist])
    top15 = pd.DataFrame(top15.most_common(15))
    top15.columns = ['Word','Frequency']
    # top15 = temp.style.background_gradient(cmap='Purples')

    #############################################################################################################################

    # ''' Word Cloud'''
    wordcloud = WordCloud(background_color='white', colormap='inferno').generate(' '.join(news_df['Text']))

    #############################################################################################################################

    # '''Bar Chart for News Articles Sentiment'''
    neu_ = []
    pos_ = []
    neg_ = []

    for row in news_df['Text']:
        if (Analyzer.polarity_scores(row)['compound']) >= 0.1:
            pos_.append(row)
        elif (Analyzer.polarity_scores(row)['compound']) <= -0.1:
            neg_.append(row)
        else:
            neu_.append(row)

    # '''Bar Charts of Pos, Neu and Neg Articles'''
    pos_df = news_df.copy(deep=True)

    # Create a column `Compound` which has the Compound Score of Sentiment
    pos_df['Compound'] = [Analyzer.polarity_scores(x)['compound'] for x in news_df['Text']]

    # Filter to have only those samples/rows which are positive, i.e. having `Compound` value greater than 0 
    pos_df = pos_df[pos_df['Compound']>0]

    # Split each news article to get the list of words in the each article
    pos_df['word_list'] = pos_df['Text'].apply(lambda x:str(x).split())

    # Count the number of words in the entire `word_list`
    top_pos = Counter([item for sublist in pos_df['word_list'] for item in sublist])
    temp_pos = pd.DataFrame(top_pos.most_common(15))
    temp_pos.columns = ['Frequent Words','count']

    # Create a new DataFrame `neg_df` which is a copy of `news_df`
    neg_df = news_df.copy(deep=True)

    # Create a column `Compound` which has the Compound Score of Sentiment
    neg_df['Compound'] = [Analyzer.polarity_scores(x)['compound'] for x in news_df['Text']]

    # Filter to have only those samples/rows which are negitive, i.e. having `Compound` value less than 0 
    neg_df = neg_df[neg_df['Compound']<0]

    # Split each news article to get the list of words in the each article
    neg_df['word_list'] = neg_df['Text'].apply(lambda x:str(x).split())

    # Count the number of words in the entire `word_list`
    top_neg = Counter([item for sublist in neg_df['word_list'] for item in sublist])
    temp_neg = pd.DataFrame(top_neg.most_common(15))
    temp_neg.columns = ['Frequent Words','count']

    #############################################################################################################################
    placeholder.empty()

    if not news_df.empty :
        options = st.selectbox('Select Option', ('','15 Most Frequent Words in All News Articles', 'Frequent Words by Sentiment', 'Word Cloud'))

    if options == '':
        st.write('`Please Select an Option`')

    if options == '15 Most Frequent Words in All News Articles' :
        # st.markdown('`SHOW TOP 15`')
        AgGrid(top15, fit_columns_on_grid_load=True, theme='blue')

    if options == 'Frequent Words by Sentiment' :
        # st.markdown('`SHOW BAR CHARTS OR TREE MAPS`')
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
        st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)

        visualize = st.radio('Select Visualization Method', ('Bar Chart', 'Tree Map'))
        
        if visualize == 'Bar Chart' :
            col1, col2 = st.columns(2)

            with col1 :
                fig = plt.figure(figsize=(11.7,8.27))
                ax = sns.barplot(x="count", y="Frequent Words", data=temp_pos)
                ax.set_title('15 most frequent words in Positive News Articles')
                st.pyplot(fig)

            with col2 :
                fig = plt.figure(figsize=(11.7,8.27))
                ax = sns.barplot(x="count", y="Frequent Words", data=temp_neg)
                ax.set_title('15 most frequent words in Negative News Articles')
                st.pyplot(fig)

        if visualize == 'Tree Map' :
            col1, col2 = st.columns(2)

            with col1 :
                fig = px.treemap(temp_pos, path=['Frequent Words'], values='count',title='Tree Of Most Frequent Positive Words')
                fig.update_layout(title_x=0.5, autosize=False,
                    width=700,
                    height=500,)
                # fig.show()
                st.plotly_chart(fig, use_container_width=True)

            with col2 :
                fig = px.treemap(temp_neg, path=['Frequent Words'], values='count',title='Tree Of Most Frequent Negative Words')
                fig.update_layout(title_x=0.5, autosize=False,
                    width=700,
                    height=500,)
                # fig.show()
                st.plotly_chart(fig, use_container_width=True)


    if options == 'Word Cloud' :
        # st.markdown('`SHOW WORD CLOUD`')
        plt.figure(figsize=(15,10))
        plt.imshow(wordcloud)
        plt.axis("off")
        # plt.title('\nWordcloud for news articles related to Bitcoin\n')
        plt.tight_layout()
        plt.show()
        st.pyplot()
#when we import hydralit, we automatically get all of Streamlit
import os
import sys
import pickle
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

app = hy.HydraApp(title='News Based Sentiment analysis for Bitcoin Price Prediction',navbar_theme='Pink')

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
    '''Top 15 Most Frequent Words'''
    Analyzer = SentimentIntensityAnalyzer()

    stop_words = stopwords.words('english')

    stop_words.append('https')
    stop_words.append('nbsp')

    #news = news_df.copy(deep=True)

    news_df['temp_list'] = news_df['Text'].apply(lambda x:str(x).split())

    news_df['Text'] = news_df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

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


@st.cache(persist=True, suppress_st_warning=True)
@app.addapp(title='Train Models')
def train_model():
    data_dir = os.path.join('.','data')
    model_dir = os.path.join('.','models')


    bnb = BernoulliNB()
    gnb = GaussianNB()
    knn = KNeighborsClassifier(algorithm='auto', metric='euclidean', n_neighbors=10, weights='distance')
    lr = LogisticRegression(C=0.0001, penalty='l1', solver='liblinear')
    rf = RandomForestClassifier(n_estimators=20, min_samples_split=3, min_samples_leaf=3, max_leaf_nodes=5, max_features='log2', max_depth=None, criterion='entropy')
    svc = SVC(kernel='rbf', gamma='scale', C=0.001)
    mlp = MLPClassifier(solver='adam', learning_rate='adaptive', hidden_layer_sizes=50, alpha=0.1, activation='relu')
    
    df = pd.read_csv(os.path.join(data_dir,'feature_set.csv'))
    
    row1, row2 = st.columns((20,1))
    with row1 :
        placeholder = st.empty()
        placeholder.dataframe(df)
        # AgGrid(df)


    col1,col2, col3 = st.columns(3)
    with col1 :
        models = st.selectbox('Choose Machine Learning Classification Algorithm', ('','Bernoulli Naive Bayes', 'Gaussian Naive Bayes', 'K-Nearest Neighbors', 
                                                                                'Logisitic Regression', 'Random Forest', 'Support Vector Machine', 'Multi-Layer Perceptron'))

    

        # if models == '' :
        #     """`Please choose an algorithm!`"""

    with col2 :
        scaling = st.selectbox('Choose a scaling techinque', ('','No Scaling', 'Standard Scaling', 'Min-Max Scaling', 'Robust Scaling'))

        if scaling == '' :
            placeholder.empty()
            with placeholder.container():
                # st.write('`Feature Set with the chosen scaling will appear here`!')
                st.markdown("<p style='text-align: center; color: orangered;'><b>Feature Set with the chosen scaling will appear here!</b></p>", unsafe_allow_html=True)
            # """`Please select scaling technique`"""

    with col3 : 
        upload_new = st.selectbox('Upload New Data', ('', 'Yes', 'No'))

    # if models == '' :
    #     """`Please choose an algorithm!`"""


    if scaling == 'No Scaling' :
        placeholder.empty()
        placeholder.dataframe(df)

    if scaling == 'Standard Scaling' :
        scaler = StandardScaler()
        df = df.copy(deep=True)
        df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])
        with row1 :
            placeholder.empty()
            placeholder.dataframe(df)
            # AgGrid(df)

    if scaling == 'Min-Max Scaling' :
        scaler = MinMaxScaler()
        df = df.copy(deep=True)
        df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])
        with row1 :
            placeholder.empty()
            placeholder.dataframe(df)
            # AgGrid(df)

    if scaling == 'Robust Scaling' :
        scaler = RobustScaler()
        df = df.copy(deep=True)
        df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])
        placeholder.empty()
        placeholder.dataframe(df)
        # AgGrid(df)
    
    R1, R2 = st.columns((20, 1))
    with R1 :
        R1P = st.empty()

    if models == 'Bernoulli Naive Bayes' :
        X = df.drop('Class', axis=1)
        y = df['Class']
        bnb.fit(X, y)
        bnb_y_pred = bnb.predict(X)
        bnb_acc = accuracy_score(y, bnb_y_pred)
        bnb_f1 = f1_score(y, bnb_y_pred)
        with R1 :
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: green;'><b>Bernoulli Naive Bayes Results</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {bnb_acc}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {bnb_f1}</b></p>", unsafe_allow_html=True)

    if models == 'Gaussian Naive Bayes' :
        X = df.drop('Class', axis=1)
        y = df['Class']
        gnb.fit(X, y)
        gnb_y_pred = gnb.predict(X)
        gnb_acc = accuracy_score(y, gnb_y_pred)
        gnb_f1 = f1_score(y, gnb_y_pred)
        with R1 :
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: green;'><b>Gaussian Naive Bayes Results</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {gnb_acc}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {gnb_f1}</b></p>", unsafe_allow_html=True)

    if models == 'K-Nearest Neighbors' :
        X = df.drop('Class', axis=1)
        y = df['Class']
        knn.fit(X, y)
        knn_y_pred = knn.predict(X)
        knn_acc = accuracy_score(y, knn_y_pred)
        knn_f1 = f1_score(y, knn_y_pred)
        with R1 :
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: green;'><b>K-Nearest Neighbors Results</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {knn_acc}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {knn_f1}</b></p>", unsafe_allow_html=True)

    if models == 'Logisitic Regression' :
        X = df.drop('Class', axis=1)
        y = df['Class']
        lr.fit(X, y)
        lr_y_pred = lr.predict(X)
        lr_acc = accuracy_score(y, lr_y_pred)
        lr_f1 = f1_score(y, lr_y_pred)
        with R1 :
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: green;'><b>Logisitic Regression Results</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {lr_acc}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {lr_f1}</b></p>", unsafe_allow_html=True)

    if models == 'Random Forest' :
        X = df.drop('Class', axis=1)
        y = df['Class']
        rf.fit(X, y)
        rf_y_pred = rf.predict(X)
        rf_acc = accuracy_score(y, rf_y_pred)
        rf_f1 = f1_score(y, rf_y_pred)
        with R1 :
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: green;'><b>Random Forest Results</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {rf_acc}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {rf_f1}</b></p>", unsafe_allow_html=True)

    if models == 'Support Vector Machine' :
        X = df.drop('Class', axis=1)
        y = df['Class']
        svc.fit(X, y)
        svc_y_pred = svc.predict(X)
        svc_acc = accuracy_score(y, svc_y_pred)
        svc_f1 = f1_score(y, svc_y_pred)
        with R1 :
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: green;'><b>Support Vector Machine Results</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {svc_acc}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {svc_f1}</b></p>", unsafe_allow_html=True)

    if models == 'Multi-Layer Perceptron' :
        X = df.drop('Class', axis=1)
        y = df['Class']
        mlp.fit(X, y)
        mlp_y_pred = mlp.predict(X)
        mlp_acc = accuracy_score(y, mlp_y_pred)
        mlp_f1 = f1_score(y, mlp_y_pred)
        with R1 :
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown('<p></p>', unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; color: green;'><b>Multi-Layer Perceptron Results</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: green;'><b>Accuracy : {mlp_acc}</b></p>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color: green;'><b>F1 Score : {mlp_f1}</b></p>", unsafe_allow_html=True)
    

    # if upload_new == 'Yes' :
    #     with col3 :
    #         uploader = st.file_uploader('Upload Feature Set', type='csv')
    #         feature_set_new = StringIO(uploader.getvalue().decode("utf-8"))

    #     with row1 :
    #         placeholder = st.empty()
    #         placeholder.dataframe(feature_set_new)

    
    col3, col4, col5 = st.columns(3)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)
    st.markdown('<p></p>', unsafe_allow_html=True)

    with col3 :
        save_button = st.button('Save Selected Model')

    with col4 :
        load_button = st.button('Load Model')

    with col5 :
        delete_button = st.button('Delete Selected Model')


#Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.
app.run()
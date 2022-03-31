import time
import requests
import pandas as pd
import numpy as np
import regex as re
import streamlit as st
from st_aggrid import AgGrid
from bs4 import BeautifulSoup
from stqdm import stqdm
from utils import remove_punctuation, batch_generator, lemmatize_text
from nltk.corpus import stopwords, wordnet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

count = 0

def create() :
    # train_options = st.selectbox('Combine with trained feature set?', ('', 'Combine with feature set models already trained on', 'Use only the newly created feature set'))
    
    col1, col2 = st.columns(2)

    global links, btc_df, count
    
    with col1 :
        try :
            links_uploader = st.file_uploader('Upload Links File', type='csv')
            if links_uploader is not None :
                LINKS = pd.read_csv(links_uploader, header=None, names=['Links'])
                links_name_input = st.text_input('Enter name to save the uploaded Links file')
                links_file_name = links_name_input + '.csv'
                if links_file_name == 'link.csv' :
                    st.error('File already exists! Please choose another name')
                if links_name_input == '' :
                    st.warning('Please enter filename for uploaded links file')
                # st.dataframe(LINKS)
                AgGrid(LINKS,fit_columns_on_grid_load=True)

        except UnboundLocalError  :
            # print('Referred before assignment')
            # st.warning('Please upload the LINKS FILE')
            st.write('')

    with col2 :
        try :
            btc_uploader = st.file_uploader('Upload Bitcoin Price File', type='csv')
            if btc_uploader is not None :
                BTC = pd.read_csv(btc_uploader, skiprows=0)
                btc_name_input = st.text_input('Enter name to save the uploaded bitcoin price file')
                btc_file_name = btc_name_input + '.csv'
                if btc_name_input == 'gemini_BTCUSD_1hr.csv' :
                    st.error('File already exists! Please choose another name')
                if btc_name_input == '' :
                    st.warning('Please enter filename for uploaded bitcoin price data file file')
                # st.dataframe(BTC)
                AgGrid(BTC,fit_columns_on_grid_load=True)

        except UnboundLocalError :
            # print('Referred before assignment')
            # st.warning('Please upload the BTC FILE')
            st.write('')

    try :
        links = LINKS
        btc_df = BTC
    except UnboundLocalError :
        st.write('')

    col3, col4, col5 = st.columns((2.5,2,1))


    with col3 :
        st.write('')

    with col4 :
        st.markdown('<p></p>', unsafe_allow_html=True)
        st.markdown('<p></p>', unsafe_allow_html=True)
        st.markdown('<p></p>', unsafe_allow_html=True)
        create_button = st.button('Create Feature Set')

    with col5 :
        st.write('')

    if create_button :

        if links_uploader is None :
            st.warning('Upload Links CSV file')
        
        if btc_uploader is None :
            st.warning('Upload Bitcoin Price Data CSV file')

        else :
            nltk.download(stopwords)
            nltk.download(wordnet)
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:66.0) Gecko/20100101 Firefox/66.0", "Accept": 
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.5", "Accept-Encoding": "gzip, deflate", "DNT": "1", "Connection": "close", "Upgrade-Insecure-Requests": "1"}

            links = links.drop_duplicates()
            indexes = links[links['Links'].str.contains('video')].index
            links = links.drop(indexes)
            
            articles = []
            dates = []

            pbar = stqdm(range(len(links)), unit='Links')
            st.write('Processing Links. This may take a while. Please be patient. Thank you 🙂')
            pbar.set_description('Links Processed')

            for link in links['Links'] :
                try :
                    start = time.time()
                    URL = link

                    page = requests.get(URL, headers=headers)
                    # st.write(page)

                    soup = BeautifulSoup(page.content, "html.parser")

                    # Bitcoin Magazine Scrapper (DONE)
                    if 'bitcoinmagazine.com' in URL :
                        try :
                            date = soup.find('time')
                            date = str(date)[:42]
                            date = re.sub('<time datetime="','',date)
                            date = date[:-1]

                            text = []
                            for data in soup.find_all('p') :
                                text.append(data.get_text())

                            if text :
                                articles.append((''.join(text)))
                                dates.append(date)
                        except :
                            continue

                    # CoinDesk Scrapper (DONE)
                    if 'coindesk.com' in URL :
                        try :
                            date = soup.find_all('span', {'class':'typography__StyledTypography-owin6q-0 dHSCiD'})
                            date = re.sub(',|at|UTC', '', date[279].text)
                            date = re.sub(' a.m.', ' AM', date)
                            date = re.sub(' p.m.', ' PM', date)
                            date = re.sub('  ', ' ', date)

                            result = soup.find_all('p')

                            text = []
                            for r in result :
                                r = str(r)
                                if (r.startswith('<p>')) and (r.endswith('</p>')) :
                                    r = re.sub('<p>|</p>|<b>|</b>|<a |</a>|</p>|<i>|</i>|<br>|</br>|<br/>|<|>','',r)
                                    text.append(r)

                            if text :
                                articles.append(''.join(text))
                                dates.append(date)
                        except :
                            continue



                    # CryptoSlate Scrapper (DONE)
                    if 'cryptoslate.com' in URL :
                        try :
                            date = soup.find('span', class_='post-date')
                            date = str(date)
                            date = re.sub('<span class="post-date">| UTC</span></span>|<span class="time">', '', date)
                            date = re.sub('at', '', date)

                            result = soup.find_all('p')
                            text = []
                            result = soup.find_all('p')
                            for r in result :
                                r = str(r)
                                r = re.sub('<p>|</p>|<span>|</span>|<p class="post-subheading">|<p class="image-credit">|<a>|</a>', '', str(r))
                                r = re.sub('<script>|</script>|[|]', '', r)
                                text.append(r)

                            if text :
                                articles.append(''.join(text))
                                contains_digit = any(map(str.isdigit, date))
                                dates.append(date)
                        except :
                            continue


                    # Yahoo Scrapper (DONE)
                    if 'yahoo.com' in URL :
                        try :
                            date = soup.find('time')
                            date = str(date)
                            date = date.replace('"','')
                            date = re.sub('<time class=caas-attr-meta-time datetime=|>|</time>', '',date)
                            date = date[:24]

                            text = []
                            result = soup.find_all('p', text=True)
                            for r in result :
                                r = str(r)
                                r = re.sub('"|</p>|<p>','', r)
                                if r.startswith('<p class=M(0) C($summaryColor) Fz(14px) Lh(1.43em) LineClamp(3,60px)>') :
                                    r = r[69:]
                                text.append(r)

                            if text :
                                articles.append(''.join(text))
                                contains_digit = any(map(str.isdigit, date))
                                dates.append(date)

                        except :
                            continue


                    # Forbes Scrapper (DONE)
                    if 'forbes' in URL :
                        try :
                            date = soup.find_all('time')
                            date = str(date)
                            date = date[:52]
                            date = date.replace('[','')
                            date = date.replace(']','')
                            date = date.replace('\\','')
                            date = re.sub('<time>|</time>|</time|,', '', date)
                            date = date[:19]

                    #         date = str(date[18]) + str(date[19])
                    #         date = re.sub('<time>|</time>','', date)
                    #         date = re.sub(',',' ', date)
                    #         date = re.sub('  ',' ',date)[:-3]

                            text = []
                            result = soup.find_all('p')
                            for r in result :
                                r = str(r)
                                r = re.sub('"|</p>|<p>','', r)
                                r = re.sub('<p class=color-body light-text>', '',r)
                                text.append(r)

                            if text :
                                articles.append(''.join(text))
                                dates.append(date)

                        except :
                            continue


                    # Nulltx Scrapper (DONE)
                    if 'nulltx.com' in URL :
                        try :
                            date = soup.find('time')
                            date = re.sub('<time class="entry-date updated td-module-date" datetime="|</time>','',str(date))
                            date = date[:25]

                            text = []
                            result = soup.find_all('p')
                            for r in result :
                                r = str(r)
                                r = re.sub('<p>|</p>|<strong>|</strong>', '', r)
                                r = re.sub('<p class="comment-form-cookies-consent"><input id="wp-comment-cookies-consent" name="wp-comment-cookies-consent" type="checkbox" value="yes"><label for="wp-comment-cookies-consent">Save my name, email, and website in this browser for the next time I comment.</label></input>','',r)
                                r = re.sub('<p class="form-submit"><input class="submit" id="submit" name="submit" type="submit" value="Post Comment"> <input id="comment_post_ID" name="comment_post_ID" type="hidden" value="103352">','',r)
                                r = re.sub('<em>Image Source: Vintage Tone/<a href="http://shutterstock.com/" rel="noopener nofollow" target="_blank">Shutterstock.com</a></em>','',r)
                                r = re.sub('<input id="comment_parent" name="comment_parent" type="hidden" value="0">|</input></input></input>','',r)
                                r = re.sub('nulltx.com is part of the Null Transaction PR media group.','',r)
                                r = r.replace('\n\n','')
                                text.append(r)

                            if text :
                                articles.append(''.join(text))
                                dates.append(date)

                        except :
                            continue

                        # st.write('NullTx', page, len(text), date)

                    
                        if 'blockonomi.com' in URL :
                            try :
                                date = soup.find('time', class_='post-date')
                                date = re.sub('<time class="post-date" datetime="|</time>','',str(date))
                                date = date[:25]

                                text = []
                                result = soup.find_all('p')
                                for r in result :
                                    r = str(r)
                                    r = re.sub('<p>|</p>|<strong>|</strong>|<br>|</br>|<br/>|<em>|</em>|<a>|</a>', '', r)
                                    r = re.sub('<p class="toc_title">|<p class="text author-bio">|<p class="copyright">','',r)
                                    text.append(r)

                                if text :
                                    articles.append(''.join(text))
                                    contains_digit = any(map(str.isdigit, date))
                                    if contains_digit :
                                        dates.append(date)

                            except :
                                continue
                
                    end = time.time()

                    pbar.update(1)

                except :
                    continue


            news_df = pd.DataFrame(data={'Date':dates, 'Text':articles})

            st.success('News Articles Collected Successfully!')

            st.info('Creating the new feature set. This may take a while. Please be patient. Thank you 🙂')

            ####################################################################################################################

            for index, date in enumerate(news_df['Date']) :
                if date == '' :
                    news_df.drop(index, inplace=True)

            indexes = []

            for index, date in enumerate(news_df['Date']) :
                contains_digit = any(map(str.isdigit, date))
                if contains_digit == False :
                    indexes.append(index)

            news_df = news_df.drop(indexes)

            news_df['Date'] = pd.to_datetime(news_df['Date'], utc=True)

            news_df['Time'] = news_df['Date'].dt.round('H').dt.time
            news_df['Date'] = news_df['Date'].dt.date

            news_df = news_df[['Date', 'Time', 'Text']]

            news_df['Date'] = news_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))

            news_df['Time'] = news_df['Time'].apply(lambda x: x.strftime('%H-%M-%S'))

            btc_df['Date'] = pd.to_datetime(btc_df['Date'], utc=True)
            btc_df['Time'] = btc_df['Date'].dt.time
            btc_df['Date'] = btc_df['Date'].dt.date

            btc_df = btc_df[['Unix Timestamp', 'Date', 'Time', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
            btc_df = btc_df.sort_values('Date', ascending=False).reset_index(drop=True)

            news_df = news_df.sort_values('Date', ascending=False).reset_index(drop=True)

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
           
            # ''' STATISTICAL DRIVIEN APPROACH'''

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

            # st.dataframe(news_df)
            # st.dataframe(btc_df)

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
            # st.dataframe(btc_df)

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

            st.success('Feature set created sucessfully!')


            # if train_options == 'Combine with feature set models already trained on':
            #     news_df = pd.concat([df, news_df], axis=1, ignore_index=True)
            #     return news_df, btc_df, feature_set

            # if train_options == 'Use only the newly created feature set':
            #     return news_df, btc_df, feature_set

            count += 1

            news_df.to_csv(links_file_name, index=False)
            btc_df.to_csv(btc_file_name, index=False)
            feature_set.to_csv('CREATED FEATURE SET' + str(count)+'.csv', index=False)

            return news_df, btc_df, feature_set
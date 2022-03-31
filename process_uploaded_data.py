import time
import requests
import pandas as pd
import streamlit as st
import regex as re
from stqdm import stqdm
from bs4 import BeautifulSoup
from utils import remove_punctuation, batch_generator, lemmatize_text
from nltk.corpus import stopwords, wordnet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def process(LINKS_DF, BTC_DF) :
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:66.0) Gecko/20100101 Firefox/66.0", "Accept": 
    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US,en;q=0.5", "Accept-Encoding": "gzip, deflate", "DNT": "1", "Connection": "close", "Upgrade-Insecure-Requests": "1"}

    articles = []
    dates = []
    count = 0

    pbar = stqdm(range(len(LINKS_DF)))
    st.write('Processing Links. This may take a while. Please be patient. Thank you 🙂')
    pbar.set_description('Links Processed')

    for link in LINKS_DF['Links'] :
        start = time.time()
        URL = link

        page = requests.get(URL, headers=headers)
        # st.write(page)

        soup = BeautifulSoup(page.content, "html.parser")
        
        # Bitcoin Magazine Scrapper (DONE)
        if 'bitcoinmagazine.com' in URL :
            date = soup.find('time')
            date = str(date)[:42]
            date = re.sub('<time datetime="','',date)
            date = date[:-1]

            text = []
            for data in soup.find_all('p') :
                text.append(data.get_text())
                
            if text :
                dates.append(date)
                articles.append((''.join(text)))

            # st.write('Bitcoin Magazine', page, len(text), date)

        # CoinDesk Scrapper (DONE)
        if 'coindesk.com' in URL :
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
                dates.append(date)
                articles.append(''.join(text))

            # st.write('CoinDesk', page, len(text), date)

        # CoinTelegraph Scrapper
        if 'cointelegraph.com' in URL :
            date = soup.find_all('span', {'class':'typography__StyledTypography-owin6q-0 dHSCiD'})
            date = re.sub(',|at|UTC', '', date[279].text)
            date = re.sub(' a.m.', ' AM', date)
            date = re.sub(' p.m.', ' PM', date)
            date = re.sub('  ', ' ', date)

            text = []
            result = soup.find_all('p', text=True)
            for r in result :
                r = str(r)
                if (r.startswith('<p>')) and (r.endswith('</p>')) :
                    r = re.sub('<p>|</p>|<b>|</b>|<a |</p>','',r)
                    text.append(r)

            if text :
                articles.append(''.join(text))
                dates.append(date)
            
            # st.write('CoinTelegraph', page, len(text), date)

        # CryptoSlate Scrapper (DONE)
        if 'cryptoslate.com' in URL :
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
                dates.append(date)

            # st.write('CryptoSlate', page, len(text), date)

        # Yahoo Scrapper (DONE)
        if 'yahoo.com' in URL :
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
                dates.append(date)

            # st.write('Yahoo', page, len(text), date)

        # Forbes Scrapper (DONE)
        if 'forbes' in URL :
            URL = link
            # Request access to the URL
            page = requests.get(URL, headers=headers)

            # Parse the html using BeautifulSoup for scrapping
            soup = BeautifulSoup(page.content, "html.parser")
            date = soup.find_all('time')
            date = str(date)
            date = date[:52]
            date = date.replace('[','')
            date = date.replace(']','')
            date = date.replace('\\','')
            date = re.sub('<time>|</time>|</time|,', '', date)
            date = date[:19]
            # print(date)
            
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

            # st.write('Forbes', page, len(text), date)

        # Nulltx Scrapper (DONE)
        if 'nulltx.com' in URL :
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

            # st.write('NullTx', page, len(text), date)

        # Blockonomi Scrapper (DONE)
        if 'blockonomi.com' in URL :
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
                dates.append(date)

            # st.write('Blockonomi', page, len(text), date)
        
        end = time.time()
        
        count += 1

        pbar.update(1)

    news_df = pd.DataFrame(data={'Date':dates, 'Text':articles})
    # st.dataframe(news_df)
    # st.dataframe(BTC_DF)

    # messages = st.empty()

    # with messages.container() :
    st.success('News Articles Collected Successfully!')

    message_container = st.empty()

    with message_container.container() :
        st.info('Processing the uploaded data. This may take a while. Please be patient. Thank you 🙂')

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

    BTC_DF['Date'] = pd.to_datetime(BTC_DF['Date'], utc=True)
    BTC_DF['Time'] = BTC_DF['Date'].dt.time
    BTC_DF['Date'] = BTC_DF['Date'].dt.date

    BTC_DF = BTC_DF[['Unix Timestamp', 'Date', 'Time', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]
    BTC_DF = BTC_DF.sort_values('Date', ascending=False).reset_index(drop=True)

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

    # news_df['temp_list'] = news_df['Text'].apply(lambda x:str(x).split())

    message_container.empty()

    return news_df, BTC_DF


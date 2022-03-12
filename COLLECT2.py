# import os
import pandas as pd
import requests
# import streamlit as st
from bs4 import BeautifulSoup
import itertools

def batch_generator(iterable, batch_size=1000):
    iterable = iter(iterable)

    while True:
        batch = list(itertools.islice(iterable, batch_size))
        if len(batch) > 0:
            yield batch
        else:
            break

links = pd.read_csv('E:\Others\Wyzant\Elly\Streamlit Website\LINKS.csv',header=None)
links = links[:10]

headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:66.0) Gecko/20100101 Firefox/66.0", "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", 
"Accept-Language": "en-US,en;q=0.5", "Accept-Encoding": "gzip, deflate", "DNT": "1", "Connection": "close", "Upgrade-Insecure-Requests": "1"}

articles = []
dates = []
scrapped_links = []

# Initialize a count variable to keep track of how many news articles have been scrapped so far
count = 0

batches = []

for batch in batch_generator(links[0], 10):
    batches.append(batch)

# Iterate over all the links in the DataFrame `links`
for batch in batches :
    for link in batch :
        URL = link
        # Request access to the URL
        page = requests.get(URL, headers=headers)
    
        # Parse the html using BeautifulSoup for scrapping
        soup = BeautifulSoup(page.content, "html.parser")

        # Find the time in the the `time` class in the html
        result = soup.find('time')

        if result == None:
        # Find the `post-time` class which is another html class that has date and time in it for some websites
            date = soup.find("li",attrs={'class':'post-time'})#.text.strip()
            
            if date == None:
                continue
            else :
                page = requests.get(link, headers=headers)#, proxies={'http':proxy, 'https':proxy})
                soup = BeautifulSoup(page.content, 'html.parser')
                text = ' '
                for data in soup.find_all("p"):
                    paragraph = data.get_text()
                    text = text + paragraph
                    print(paragraph)
                articles.append(text)
                scrapped_links.append(link)

        else :
            for i in soup.findAll('time'):
                if i.has_attr('datetime'):
                    time = i['datetime']
                    #print(time)

            # Begin scrapping to get the text of the news articles
            page = requests.get(link, headers=headers)#, proxies={'http':proxy, 'https':proxy})
            soup = BeautifulSoup(page.content, 'html.parser')
            # Set text of the article to empty string
            text = ' '

            # Find all the paragraphs in the article
            for data in soup.find_all("p"):

                # Get the text from the paragraph
                paragraph = data.get_text()
                # Add the text for the paragraph by concetanting it with `text`
                text = text + paragraph
                print(text)
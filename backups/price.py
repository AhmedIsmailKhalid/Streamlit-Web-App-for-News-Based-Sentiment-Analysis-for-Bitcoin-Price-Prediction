from bs4 import BeautifulSoup 
import requests 
import time



coin = 'bitcoin'

#Set the last price to negative one
last_price = -1
#Create an infinite loop to continuously show the price
while True:
#Choose the cryptocurrency that you want to get the price of (e.g. bitcoin, litecoin)
    crypto = 'bitcoin' 
    #Get the price of the crypto currency
    
    url = "https://www.google.com/search?q="+coin+"+price"

    #Make a request to the website
    HTML = requests.get(url) 

    #Parse the HTML
    soup = BeautifulSoup(HTML.text, 'html.parser') 

    #Find the current price 
    #text = soup.find("div", attrs={'class':'BNeawe iBp4i AP7Wnd'}).text
    price = soup.find("div", attrs={'class':'BNeawe iBp4i AP7Wnd'}).find("div", attrs={'class':'BNeawe iBp4i AP7Wnd'}).text
    
    #Check if the price changed
    if price != last_price:
        print(crypto+' price: ',price) #Print the price
        last_price = price #Update the last price
    time.sleep(3) #Suspend execution for 3 second
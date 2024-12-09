# script to scrape the fox news front page for headlines
# and store them in a text file

import requests
import selenium
from bs4 import BeautifulSoup

# selenium initialize
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options



def scrape_fox():

    response = requests.get("https://www.foxnews.com/")
    # create a BeautifulSoup object
    soup = BeautifulSoup(response.content, "html.parser")


    headlines = soup.find_all("h3", class_="title")
    # print the headlines to the console

    with open("fox_headlines.txt", "w") as f:
        for headline in headlines:
            # clean headline of whitespace and newlines
            headline = headline.text.strip()
            f.write(headline + "\n")

def scrape_cnn():
    response = requests.get("https://www.cnn.com/")

    soup = BeautifulSoup(response.content, "html.parser")

    headlines = soup.find_all("span", class_="container__headline-text")

    with open("cnn_headlines.txt", "w") as f:
        for headline in headlines:
            headline = headline.text.strip()
            f.write(headline + "\n")


scrape_fox()
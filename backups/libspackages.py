# Importing Neccessary Libraries
import pandas as pd
import numpy as np
import requests
import regex as re
from bs4 import BeautifulSoup
import math
import json
import time

# NLP Imports
import string
import nltk
from nltk.corpus import stopwords, wordnet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, RegexpTokenizer

# Visualization and EDA libraries
from collections import Counter, OrderedDict
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import plotly.express as px

# Machine Learning Library Imports
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier, VotingRegressor

# Deep Learning Library Imports
import tensorflow as tf
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Model
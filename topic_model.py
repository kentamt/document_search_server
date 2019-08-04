import time
import logging
from collections import Counter

import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd

# Natural Language Processing Libraries
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import datasets

# Vizualization
from wordcloud import WordCloud

class TopicModel:
    def __init__(self):
        # member variables
        self.df = None  # DataFrame for original data ?
        self.texts = None # 
        self.stopwords = None # TODO: install stopwords by nltk 
        pass

    def add_texts(self, texts):
        pass

    def _preprocess(self, texts):
        """
        1. remove stopwords
        2. lemmatization
        """
        # stop_words = stopwords.words('english')
        toks = gensim.utils.simple_preprocess(str(text), deacc=True)
        wn = WordNetLemmatizer()
        return [wn.lemmatize(tok, simplify(pos)) for tok, pos in nltk.pos_tag(toks) if tok not in stop_words]

    def train(self, eta):
        pass

    def upadte(self, new_texts):
        pass 
    
    def save_model(self, path):
        pass 

    def load_model(self, path):
        pass 
    
    def recommend(self):
        pass

    def get_unused_texts(self):
        pass

    def get_used_texts(self):
        pass 

if __name__ == "__main__":
    # df = pd.read_csv("./arxivs_data.csv")
    # print(df.head())
    print("OK")
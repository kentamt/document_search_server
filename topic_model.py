# 
# @file topic_mode.py
# 
# @brief 
#
# @author Kenta Matsui
# @date 3-Aug. 2019
# 

import time
import logging
from collections import Counter

import numpy as np 
from matplotlib import pyplot as plt 
import pandas as pd

# Natural Language Processing Libraries
import gensim
from gensim import similarities
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
        self.docs = None 
        self.corpus = None
        self.stopwords = None # TODO: install stopwords by nltk 
        self.lda = None
        self.num_topics = None

        # display training logs
        logging.basicConfig(format='%(message)s', level=logging.INFO)

    def add_texts(self, texts):
        pass

    def docs_to_texts(self, docs):
        """
        @param docs: [str, str, str, ..., str]
        @return texts: [[word, word, ... , ], [],[]...,[]]
        """
        # return [[w for w in doc.lower().split() if w not in self.stopwords] for doc in docs]
        return [[w for w in doc.lower().split()] for doc in docs]
    
    def create_corpus_from_doc(self, doc):

        text = self.preprocess(doc)
        corpus = self.dictionary.doc2bow(text)
        return corpus

    def create_corpus_from_df(self, df):
        """
        function for test
        """
        num_docs = 10  # how many documents to use

        self.docs = [e for e in df.loc[0:num_docs, "abstract"]] # df has "abstract" column

        # self.texts = self.docs_to_texts(self.docs) # old ver
        self.texts = [self.preprocess(line) for line in self.docs] # remove stop words and lemmatization

        self.dictionary = gensim.corpora.Dictionary(self.texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]

    # def split_test_corpus(self, ratio):
    #     """
    #     TODO: randomize
    #     TODO: throw exception
    #     """
    #     if 0 < ratio < 1:
    #         printf("ratio must be 0 to 1")
    #         return None, None
        
    #     test_size = int(len(self.corpus) * ratio)
    #     test_corpus = self.corpus[:test_size]
    #     train_corpus = self.corpus[test_size:]

    #     return train_corpus, test_corpus

    def load_nltk_data(self):
        """
        初回起動時だけnltkのデータを読み込む。結構時間かかる
        """
        nltk.download() # TODO: 人の操作が必要かも。もともとDBとかに持てないかな

    def _simplify(self, penn_tag):
        pre = penn_tag[0]
        if (pre == 'J'):
            return 'a'
        elif (pre == 'R'):
            return 'r'
        elif (pre == 'V'):
            return 'v'
        else:
            return 'n'

    def preprocess(self, text):
        """
        1. remove stopwords
        2. lemmatization
        """
        stop_words = stopwords.words('english')
        toks = gensim.utils.simple_preprocess(str(text), deacc=True)
        wn = WordNetLemmatizer()
        return [wn.lemmatize(tok, self._simplify(pos)) for tok, pos in nltk.pos_tag(toks) if tok not in stop_words]

    def create_eta(self, priors, etadict, num_topics):
        """
        TODO: write functions to create priors
        """
        eta = np.full(shape=(num_topics, len(etadict)), fill_value=1) # create a (ntopics, nterms) matrix and fill with 1
        
        for word, topic in priors.items(): # for each word in the list of priors
            keyindex = [index for index,term in etadict.items() if term==word] # look up the word in the dictionary
        
            if (len(keyindex)>0): # if it's in the dictionary
                eta[topic,keyindex[0]] = 1e7  # put a large number in there

        eta = np.divide(eta, eta.sum(axis=0)) # normalize so that the probabilities sum to 1 over all topics
        return eta 

    def set_num_topics(self, num_topics):
        self.num_topics = num_topics

    def train(self, eta="auto", num_pass=10):
        """
        TODO: throw exception
        """        

        if self.corpus is None:
            print("corpus does not exist.")
            return -1
        if self.dictionary is None:
            print("dictionary does not exist.")
            return -1
        if self.num_topics is None:
            print("num topics is not difined.")
            return -1

        self.lda = gensim.models.ldamodel.LdaModel(
            corpus=self.corpus,
            num_topics=self.num_topics,
            id2word=self.dictionary,
            passes=num_pass,
            eta = eta
        )

        return 1

    def vizualize_result(self, figx=30, figy=30):
        plt.figure(figsize=(figx, figy))

        for t in range(self.lda.num_topics):
            plt.subplot(int(self.lda.num_topics/2), 2, t+1)
            x = dict(self.lda.show_topic(t,200))
            im = WordCloud().generate_from_frequencies(x)
            plt.imshow(im)
            plt.axis("off")
            plt.title("Topic #" + str(t))

    def disp_topic_words(self, topic_id):
        for t in self.lda.get_topic_terms(topic_id):
            print(" - {}: {}".format(self.dictionary[t[0]], t[1]))

    def calc_topic_distribution(self, doc):
        test_corpus = self.create_corpus_from_doc(doc)
        return sorted(self.lda.get_document_topics(test_corpus), key=lambda t:t[1], reverse=True)

    def disp_topic_distribution(self, doc):
        for t in self.calc_topic_distribution(doc):
            print("Topic {}: {}".format(t[0], t[1]))

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
    df = pd.read_csv("./arxivs_data.csv")

    topic_model = TopicModel()
    topic_model.set_num_topics(5)
    topic_model.create_corpus_from_df(df)
    topic_model.train(num_pass=1)
    topic_model.disp_topic_words(1)
    
    doc  = df.iloc[-1]["abstract"]
    topic_model.disp_topic_distribution(doc)

    print("OK")
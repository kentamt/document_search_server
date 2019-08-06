# 
# @file topic_mode.py
# 
# @brief 
#
# @author Kenta Matsui
# @date 3-Aug. 2019
# 

import pickle

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


class Data:
    def __init__(self, texts, docs, corpuses, flags):
        self.texts = texts
        self.docs = docs
        self.corpuses = corpuses
        self.is_trained = flags

    def get(self, ind):
        return self.texts[ind], self.docs[ind], self.corpuses[ind], self.is_trained[ind]

    def get_doc(self, ind):
        return self.docs[ind]

    def get_text(self, ind):
        return self.texts[ind]

    def get_corpus(self, ind):
        return self.corpuses[ind]

    def get_is_trained(self, ind):
        return self.is_trained[ind]

    def add(self, text, doc, corpus, flag):
        self.docs.append(doc)
        self.texts.append(text)
        self.corpuses.append(corpus)
        self.is_trained.append(flag)

    def set_trained(self, ind):
        self.is_trained[ind] = True

    def reset_trained(self, ind):
        self.is_trained[ind] = False

class TopicModel:
    def __init__(self):
        # member variables
        self.df = None  # DataFrame for original data ?
        
        self.texts = None
        self.docs = None 
        self.corpus = None
        self.trained_flags = None

        self.stopwords = None # TODO: install stopwords by nltk 
        self.lda = None
        self.num_topics = None
        self.train_docs_ids = None
        self.unknown_docs_ids = None
        self.liked_doc_ids = None 

        # display training logs
        logging.basicConfig(format='%(message)s', level=logging.INFO)

    def get_doc(self, ind):
        return self.docs[ind]    

    def add_doc(self, doc):
        text = self.preprocess(doc)
        curp = self.dictionary.doc2bow(text) # 辞書は学習につかう単語だけのものとする。レコメンドのときは既存の辞書をつかう
        
        self.docs.append(doc)
        self.texts.append(text)
        self.corpus.append(curp)
        self.trained_flags.append(False)
    
    def add_docs(self, docs):
        """
        TODO: 複数の文献を一気に追加する
        """
        pass


    def docs_to_texts(self, docs):
        """
        @param docs: [str, str, str, ..., str]
        @return texts: [[word, word, ... , ], [],[]...,[]]
        """
        # return [[w for w in doc.lower().split() if w not in self.stopwords] for doc in docs]
        return [[w for w in doc.lower().split()] for doc in docs]
    
    def corpus_from_doc(self, doc):

        text = self.preprocess(doc)
        corpus = self.dictionary.doc2bow(text)
        return corpus

    def update_dictionary(self, texts):
        """
        trainingに使う単語の辞書
        """
        return gensim.corpora.Dictionary(texts)        

    def create_corpus_from_df(self, df):
        """
        function for test
        TODO: use append instead of init list
        """
        num_docs = 100  # how many documents to use

        self.docs = [e for e in df.loc[0:num_docs, "abstract"]] # df has "abstract" column

        # self.texts = self.docs_to_texts(self.docs) # old ver
        self.texts = [self.preprocess(line) for line in self.docs] # remove stop words and lemmatization
        
        self.dictionary = self.update_dictionary(self.texts)

        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        self.trained_flags = [False] * num_docs
        self.liked_doc_ids = [False] * num_docs

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

        # update flags
        self.trained_flags = [ e + True for e in self.trained_flags]

        return 1

    def update_lda(self):
        """
        update LDA with new docs and trained 
        """
        pass

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

    def calc_topic_distribution_from_doc(self, doc):
        test_corpus = self.corpus_from_doc(doc)
        return sorted(self.lda.get_document_topics(test_corpus), key=lambda t:t[1], reverse=True)

    def disp_topic_distribution(self, doc):
        for t in self.calc_topic_distribution_from_doc(doc):
            print("Topic {}: {}".format(t[0], t[1]))

    def upadte(self, new_texts):
        pass 
    
    def save_model(self, path):
        pass 

    def load_model(self, path):
        pass 
    
    def set_topic_distribution_index(self):
        # create document index for all topic distribution        
        train_topic_distributions = self.lda.get_document_topics(self.corpus)
        self.doc_index = similarities.docsim.MatrixSimilarity(train_topic_distributions)

    def recommend(self, doc, num_recommended_docs = 3):
        """
        The class similarities.MatrixSimilarity is only appropriate when the whole set of vectors fits into memory. 
        For example, a corpus of one million documents would require 2GB of RAM in a 256-dimensional LSI space, when used with this class.
        Without 2GB of free RAM, you would need to use the similarities.
        Similarity class. This class operates in fixed memory, by splitting the index across multiple files on disk, 
        called shards. It uses similarities.MatrixSimilarity and similarities.
        SparseMatrixSimilarity internally, so it is still fast, although slightly more complex.
        """ 
        
        self.set_topic_distribution_index()

        # TODO: impl
        # save
        # load
        
        # TODO: あらたにdocをつくるか、corpusにマージしたあと、curpus上からidで引いてくるか要検討
        # get topic distribution for new document
        topic_dist = self.calc_topic_distribution_from_doc(doc)
        # TODO: add this topic distribution to use later
        # code

        # get similarity from all training corpus
        similar_corpus_id = self.doc_index.__getitem__(topic_dist)
        similar_corpus_id = sorted(enumerate(similar_corpus_id), key=lambda t: t[1], reverse=True) 
        recommended_docs_ids = [ e[0] for e in similar_corpus_id[:num_recommended_docs]]
        print (recommended_docs_ids)
        return recommended_docs_ids
        

    def get_unused_texts(self):
        pass

    def get_used_texts(self):
        pass 

if __name__ == "__main__":

    use_pickle = True

    df = pd.read_csv("./arxivs_data.csv")

    if not use_pickle:
        topic_model = TopicModel()
        topic_model.set_num_topics(5)
        topic_model.create_corpus_from_df(df)
        topic_model.train(num_pass=1)
        with open("./topic_model.pickle", "wb") as f:
            pickle.dump(topic_model, f)

    with open("./topic_model.pickle", "rb") as f:
        topic_model = pickle.load(f)
        # topic_model.disp_topic_words(1)    
        doc  = df.iloc[-1]["abstract"]
        topic_model.disp_topic_distribution(doc)
        for ind in topic_model.recommend(doc):
            print(topic_model.get_doc(ind))
            print("---")

    
    

    print("Done")
# 
# @file topic_mode.py
# 
# @brief 
# 
# @author Kenta Matsui
# @version 0.9.0
# @date 3-Aug. 2019
# @copyright Copyright (c) 2019
# 


import time
from datetime import datetime
import pickle
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

# Vizualization
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim


class TopicModel:
    
    def __init__(self):

        # member variables
        
        # for better debug
        self.df = None  # DataFrame for original data ?
        
        # data
        self.corpus = None
        self.doc_ids = None
        self.trained_flags = None
        self.train_docs_ids = None
        self.unknown_docs_ids = None
        self.liked_doc_ids = None 
        self.stopwords = None # TODO: install stopwords by nltk 

        # model   
        self.lda = None
        
        # model parameters
        self.num_topics = None

        # model info
        self.model_create_datetime = None
        self.num_docs = 0

        # run options
        self.vis = None

        # display training logs
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        # pyLDAvis.enable_notebook()

    def get_model_info(self):
        ret = [self.num_topics, self.num_docs, self.model_create_datetime]
        # self.perplexity # TODO
        return ret

    def add_doc(self, doc):
        """
        """
        text = self.preprocess(doc)
        curp = self.dictionary.doc2bow(text)
        
        self.corpus.append(curp)
        self.trained_flags.append(False)
    
        self.num_docs += 1

    def add_docs(self, docs):
        """
        """        
        num_docs = len(docs)
        
        texts = [self.preprocess(doc) for doc in docs] # remove stop words and lemmatization
        self.corpus.extend([self.dictionary.doc2bow(text) for text in texts])
        self.trained_flags.extend([False] * num_docs)

        self.num_docs += num_docs

    def corpus_from_doc(self, doc):

        text = self.preprocess(doc)
        corpus = self.dictionary.doc2bow(text)
        return corpus

    def update_dictionary(self, texts):
        """
        ! Caution
        ! you must use the same dictionary (mapping between words and their integer ids) for both training, updates and inference.
        ! Which means you can update the model with new documents, but not with new word types.

        ! Check out the HashDictionary class which uses the "hashing trick" to work around this limitation (but the hashing trick comes with its own caveats).
        """
        return gensim.corpora.Dictionary(texts)        

    def create_corpus_from_df(self, df):
        """
        function for test
        TODO: use append instead of init list
        """
        num_docs = 3000  # how many documents to use
        docs = [e for e in df.loc[0:num_docs, "abstract"]] # df has "abstract" column
        texts = [self.preprocess(doc) for doc in docs] # remove stop words and lemmatization

        # Create new dictionary
        self.dictionary = self.update_dictionary(texts)

        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        self.trained_flags = [False] * num_docs

        self.num_docs = num_docs
        self.doc_ids = [e for e in df.index]


    def load_nltk_data(self):
        """
        init nltk data. It should be done when the first time.
        """
        nltk.download("stopwords") 
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        self.wn = WordNetLemmatizer()
        # nltk.download("stopwords")         
        # nltk.download_shell()        
        # nltk.data.path.append("/Users/MiniBell/workspace/sazanami/nltk_data")
        # print(nltk.data.path)
        

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
        ret = [self.wn.lemmatize(tok, self._simplify(pos)) for tok, pos in nltk.pos_tag(toks) if tok not in stop_words]
        return ret

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

    def train(self, eta="auto", alpha="auto", num_pass=10):
        """
        @ret -1: error
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
        self.trained_flags = [ True for e in self.trained_flags]

        # set all topic distoributions
        self.set_topic_distribution_index()

        # set current datetime
        self.model_create_datetime = datetime.now()

        return 1

    def update_lda(self):
        """
        update LDA with new docs and trained 
        
        ! Caution
        ! you must use the same dictionary (mapping between words and their integer ids) for both training, updates and inference.
        ! Which means you can update the model with new documents, but not with new word types.

        ! Check out the HashDictionary class which uses the "hashing trick" to work around this limitation (but the hashing trick comes with its own caveats).
        """

        new_corpus = [e for f, e in zip(self.trained_flags, self.corpus) if not f]

        # new_docs = [e for f, e in zip(self.trained_flags, self.docs) if not f]
        # for doc in new_docs:
        #     print(doc)
        #     print("----")
        
        # update Model
        self.lda.update(new_corpus)

        # update flags
        self.trained_flags = [True for e in self.trained_flags]
        
        # set all topic distoributions
        self.set_topic_distribution_index()

    def vizualize_result(self, figx=30, figy=30):
        plt.figure(figsize=(figx, figy))

        for t in range(self.lda.num_topics):
            plt.subplot(np.ceil(self.lda.num_topics/2), 2, t+1)
            x = dict(self.lda.show_topic(t,200))
            im = WordCloud().generate_from_frequencies(x)
            plt.imshow(im)
            plt.axis("off")
            plt.title("Topic #" + str(t))
        plt.show()

    def save_lda_vis_as_html(self):
        vis = pyLDAvis.gensim.prepare(self.lda, self.corpus, self.dictionary, sort_topics=False)
        pyLDAvis.save_html(vis, './pyldavis_output.html')

    def get_topic_terms(self, topic_id):
        return self.lda.get_topic_terms(topic_id)

    def disp_topic_words(self, topic_id):
        for t in self.lda.get_topic_terms(topic_id):
            print(" - {}: {}".format(self.dictionary[t[0]], t[1]))

    def calc_topic_distribution_from_doc(self, doc):
        test_corpus = self.corpus_from_doc(doc)
        return sorted(self.lda.get_document_topics(test_corpus), key=lambda t:t[1], reverse=True)

    def calc_topic_distribution_from_corpus(self, corpus):
        return sorted(self.lda.get_document_topics(corpus), key=lambda t:t[1], reverse=True)

    def disp_topic_distribution(self, doc):
        for t in self.calc_topic_distribution_from_doc(doc):
            print("Topic {}: {}".format(t[0], t[1]))


    def save_model(self, path):
        print("Not implimented yew")
        pass 

    def load_model(self, path):
        print("Not implimented yew")
        pass 
    
    def set_topic_distribution_index(self):
        # create document index for all topic distribution        
        # self.topic_distributions = self.lda.get_document_topics(self.corpus)

        self.create_topic_distributions_from_curposes()
        self.doc_index = similarities.docsim.MatrixSimilarity(self.topic_distributions)

    def create_topic_distributions_from_curposes(self):
        """
        """
        self.topic_distributions = self.lda.get_document_topics(self.corpus, minimum_probability=0)
        # all_topics_csr = gensim.matutils.corpus2csc(self.topic_distributions)
        # all_topics_numpy = all_topics_csr.T.toarray()


    def get_corpus_from_id(self, ind):
        
        try:
            target_ind = self.doc_ids.index(ind)
            ret = self.corpus[target_ind]
        except:
            ret = -1

        return ret

    def recommend_from_doc(self, doc, num_similar = 3):
        """
        ! Caution
        ! The class similarities.MatrixSimilarity is only appropriate when the whole set of vectors fits into memory. 
        ! For example, a corpus of one million documents would require 2GB of RAM in a 256-dimensional LSI space, when used with this class.
        ! Without 2GB of free RAM, you would need to use the similarities.
        ! Similarity class. This class operates in fixed memory, by splitting the index across multiple files on disk, 
        ! called shards. It uses similarities.MatrixSimilarity and similarities.
        ! SparseMatrixSimilarity internally, so it is still fast, although slightly more complex.
        """ 

        # get topic distribution for new document
        topic_dist = self.calc_topic_distribution_from_doc(doc)

        # get similarity from all training corpus
        similar_corpus_id = self.doc_index.__getitem__(topic_dist)

        # sort ids by similarity
        similar_corpus_id = sorted(enumerate(similar_corpus_id), key=lambda t: t[1], reverse=True) 
        
        recommended_docs_ids = [ e[0] for e in similar_corpus_id[:num_similar]]
        
        return recommended_docs_ids  

    def recommend_from_id(self, ind, num_similar = 3):
        """
        ! Caution
        ! The class similarities.MatrixSimilarity is only appropriate when the whole set of vectors fits into memory. 
        ! For example, a corpus of one million documents would require 2GB of RAM in a 256-dimensional LSI space, when used with this class.
        ! Without 2GB of free RAM, you would need to use the similarities.
        ! Similarity class. This class operates in fixed memory, by splitting the index across multiple files on disk, 
        ! called shards. It uses similarities.MatrixSimilarity and similarities.
        ! SparseMatrixSimilarity internally, so it is still fast, although slightly more complex.
        
        TODO: 学習データのIDを指定すると、そのIDを当然推薦してくるので、それを除くこと。
        """ 
        
        # get topic distribution 
        test_corpus = self.get_corpus_from_id(ind)
        if test_corpus == -1:
            print("Doc does not exist")
            return -1

        topic_dist = self.calc_topic_distribution_from_corpus(test_corpus)

        # get similarity from all training corpus
        similar_corpus_id = self.doc_index.__getitem__(topic_dist)

        # sort ids by similarity
        similar_corpus_id = sorted(enumerate(similar_corpus_id), key=lambda t: t[1], reverse=True) 
        
        recommended_docs_ids = [ e[0] for e in similar_corpus_id[:num_similar]]
        
        return recommended_docs_ids  


    def get_unused_texts(self):
        print("Not implimented yew")
        pass

    def get_used_texts(self):
        print("Not implimented yew")
        pass 

if __name__ == "__main__":

    use_pickle = False

    # read test data
    df = pd.read_csv("./arxivs_data.csv")

    if not use_pickle:
        topic_model = TopicModel()
        topic_model.load_nltk_data()

        topic_model.set_num_topics(5)
        topic_model.create_corpus_from_df(df)

        topic_model.train(num_pass=1)
        with open("./topic_model.pickle", "wb") as f:
            pickle.dump(topic_model, f)

    with open("./topic_model.pickle", "rb") as f:
        
        topic_model = pickle.load(f)
        topic_model.load_nltk_data() # TODO: if use pickle, nltk_data dir is not set...

        # show topics
        # print("Show topics ==================================================")
        # topic_model.vizualize_result()
        
        # dump topic words
        # print("Dump topic words =============================================")
        # topic_id = 1
        # topic_model.disp_topic_words(topic_id)    
        
        # recommend docs
        print("Recommend docs ===============================================")
        doc  = df.iloc[-1]["abstract"]
        # print(doc)
        # topic_model.disp_topic_distribution(doc)

        # recommended_ids = topic_model.recommend_from_doc(doc)
        recommended_ids = topic_model.recommend_from_id(1000)
        for ind in recommended_ids:
            # print(topic_model.get_doc(ind))
            print("[recommend ]" + str(ind))


        # add new doc and update model
        # print("Add doc ======================================================")
        # topic_model.add_doc(doc)
        # topic_model.update_lda()
        
        # add new docs and update model
        # print("Add docs =====================================================")
        # docs  = df.iloc[-5:-1]["abstract"]
        # topic_model.add_docs(docs)
        # topic_model.update_lda()

        # save LDAvis
        print("Save html ====================================================")
        topic_model.save_lda_vis_as_html()

        import webbrowser
        uri = 'file://' + '/Users/MiniBell/workspace/sazanami/pyldavis_output.html'
        webbrowser.open_new_tab(uri)

    print("Done")
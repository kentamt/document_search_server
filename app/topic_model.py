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

import sys
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
from gensim.models import CoherenceModel
from gensim.test.utils import get_tmpfile,  common_texts, common_dictionary, common_corpus
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Vizualization
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim

# Profiler
# from line_profiler import LineProfiler

# Local
from error_definition import Result

class TopicModel:
    
    def __init__(self, logger_level=logging.INFO):

        # member variables
        
        # for better debug
        self.df = None  # DataFrame for original data ?
        
        # data
        self.corpuses = None
        self.doc_ids = None
        self.trained_flags = None
        self.train_docs_ids = None
        self.unknown_docs_ids = None
        self.liked_doc_ids = None 
        self.stopwords = None # TODO: install stopwords by nltk 

        # model   
        self.lda = None
        self.is_model_trained = False

        # model parameters
        self.num_topics = None

        # model info
        self.model_create_datetime = None
        self.num_docs = 0

        # topic distoribution indexes
        self.doc_index_similarity = None

        # run options
        self.vis = None

        # display training logs
        logging.basicConfig(format='%(message)s', level=logger_level)
        
        # pyLDAvis.enable_notebook()

    def get_model_info(self):
        try:
            ret = {}
            ret["num_topics"] = self.num_topics
            ret["num_docs"] = self.num_docs
            ret["date"] = self.model_create_datetime
            # self.perplexity # TODO
            return ret
        except:
            return Result.SOMETHING_WRONG

    def add_doc(self, doc, idx=None):
        """
        """

        if self.lda is None:
            return Result.NO_MODEL

        # create new id
        if idx is None:
            idx = np.max(self.doc_ids) + 1            

        # check if there is same id
        if idx in self.doc_ids:
            print("[Error] There is the same ID in corpus.")
            return Result.SAME_DOC

        else:
            print("Add new document on corpus and topic distoribution indecies.")
            text = self._preprocess(doc)
            corpus = self.dictionary.doc2bow(text)
            
            self.texts.append(text)
            self.corpuses.append(corpus)
            self.trained_flags.append(False)
            self.doc_ids.append(idx)
            self.num_docs += 1

            # push topic_distoribution_index
            topic_dist = self.lda.get_document_topics([corpus], minimum_probability=0)

            # add more documents in corpus
            self.doc_index_similarity.add_documents(topic_dist)  

            # update current datetime
            self.model_create_datetime = datetime.now()

            return Result.SUCCESS

    # def add_docs(self, docs):
    #     """
    #     """        
    #     num_docs = len(docs)
        
    #     texts = [self._preprocess(doc) for doc in docs] # remove stop words and lemmatization
    #     self.corpuses.extend([self.dictionary.doc2bow(text) for text in texts])
    #     self.trained_flags.extend([False] * num_docs)

    #     self.num_docs += num_docs

    def corpus_from_doc(self, doc):
        """
        """
        text = self._preprocess(doc)
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

    def create_corpus_from_df(self, df, num_docs = 3000):
        """
        function for test
        TODO: use append instead of init list
        """
        print("docs from df")
        docs = [e for e in df.loc[0:num_docs, "abstract"]] # df has "abstract" column
        print("texts from docs")
        texts = [self._preprocess(doc) for doc in docs] # remove stop words and lemmatization 
        self.texts = texts # NOTE: for calc coherence

        # Create new dictionary
        print("create fictionary")
        self.dictionary = self.update_dictionary(texts)

        print("create corpus")
        self.corpuses = [self.dictionary.doc2bow(text) for text in texts]
        self.trained_flags = [False] * num_docs
        self.num_docs = num_docs
        self.doc_ids = [e for e in df.loc[0:num_docs].index]

    def create_corpus_from_csv(self, filename, chunksize=10, num_docs=30):
        """
        read csv file 
        """

        reader = pd.read_csv(filename, chunksize=chunksize)
        texts = []
        doc_ids = []
        count = 0        
        for df in reader:
            print("[INFO ] read " + str(count))
            docs = [e for e in df["abstract"]] # df has "abstract" column
            doc_ids.extend([e for e in df.index])
            texts.extend([self._preprocess(doc) for doc in docs]) # remove stop words and lemmatization 
            count = count + chunksize
            if count >= num_docs:
                print("[INFO ] Reach max num docs")
                break
        
        self.texts = texts # NOTE: for calc coherence
        self.doc_ids = doc_ids
        print("[INFO ] len of texts: " + str(len(texts)))        
        
        # Create new dictionary
        print("[INFO ] create fictionary")
        self.dictionary = self.update_dictionary(texts)

        print("[INFO ] create corpus")
        self.corpuses = [self.dictionary.doc2bow(text) for text in texts]
        self.trained_flags = [False] * num_docs
        self.num_docs = num_docs

    def create_corpus_from_test_data(self):
        self.texts = common_texts
        self.corpuses = common_corpus
        self.dictionary = common_dictionary

        self.num_docs = len(self.corpuses)
        self.doc_ids = list(range(self.num_docs))
        self.trained_flags = [False] * self.num_docs

    def load_nltk_data(self, should_download=True):
        """
        init nltk data. It should be done when the first time.
        """
        if should_download:
            nltk.download("stopwords") 
            nltk.download('wordnet')
            nltk.download('averaged_perceptron_tagger')
        self.wn = WordNetLemmatizer()        

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

    def _preprocess(self, text):
        """
        1. remove stopwords
        2. lemmatization
        """
        stop_words = stopwords.words('english')
        toks = gensim.utils.simple_preprocess(str(text), deacc=True)
        ret = [self.wn.lemmatize(tok, self._simplify(pos)) for tok, pos in nltk.pos_tag(toks) if tok not in stop_words]
        return ret

    def _create_eta(self, priors, etadict, num_topics):
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

    def calc_perplexity(self):
        perplexity = np.exp2(-self.lda.log_perplexity(self.corpuses))
        return perplexity

    def calc_coherence(self, method="c_v"):
        """
        !Caution: it may take a long time
        """
        coherence_model_lda = CoherenceModel(model=self.lda, texts=self.texts, dictionary=self.dictionary, coherence=method)
        coherence_lda = coherence_model_lda.get_coherence()
        return coherence_lda

    def train(self, eta="auto", alpha="auto", num_pass=10):
        """
        """        

        if self.corpuses is None:
            print("corpus does not exist.")
            return Result.NO_CORPUS 

        if self.dictionary is None:
            print("dictionary does not exist.")
            return Result.NO_DICTIONARY

        if self.num_topics is None:
            print("num topics is not difined.")
            return Result.NO_NUM_TOPICS

        self.lda = gensim.models.ldamodel.LdaModel(
            corpus=self.corpuses,
            num_topics=self.num_topics,
            id2word=self.dictionary,
            passes=num_pass,
            eta = eta,
            # added
            random_state=100,
            update_every=1,
            chunksize=100,
            alpha='auto',
            per_word_topics=True
        )

        # update flags
        self.trained_flags = [ True for e in self.trained_flags]
        self.is_model_trained = True

        # set all topic distoributions
        self.set_topic_distribution_index()
        
        # update current datetime
        self.model_create_datetime = datetime.now()

        return Result.SUCCESS

    def update_lda(self):
        """
        update LDA with new docs and trained 
        
        ! Caution
        ! you must use the same dictionary (mapping between words and their integer ids) for both training, updates and inference.
        ! Which means you can update the model with new documents, but not with new word types.

        ! Check out the HashDictionary class which uses the "hashing trick" to work around this limitation (but the hashing trick comes with its own caveats).
        """

        new_corpus = [e for f, e in zip(self.trained_flags, self.corpuses) if not f]

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

    def save_lda_vis_as_html(self, method=None):
        if method is None:
            vis = pyLDAvis.gensim.prepare(self.lda, self.corpuses, self.dictionary, sort_topics=False)
        else:
            vis = pyLDAvis.gensim.prepare(self.lda, self.corpuses, self.dictionary, mds=method, sort_topics=False)
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
        """
        """
        self.create_topic_distributions_from_curposes()
        index_tmpfile = get_tmpfile('index')
        self.doc_index_similarity = similarities.Similarity(index_tmpfile, self.topic_distributions, num_features=len(self.dictionary))

    def set_topic_distribution_matrix(self):
        """
        """
        self.create_topic_distributions_from_curposes()
        self.doc_mat_similarity = similarities.docsim.MatrixSimilarity(self.topic_distributions)

    def set_topic_distribution_own_matrix(self):
        # create document index for all topic distribution        
        # self.topic_distributions = self.lda.get_document_topics(self.corpuses)

        # self.create_topic_distributions_from_curposes()
        # self.doc_mat_similarity = similarities.docsim.MatrixSimilarity(self.topic_distributions)
        pass
    

    def create_topic_distributions_from_curposes(self):
        """
        """
        self.topic_distributions = self.lda.get_document_topics(self.corpuses, minimum_probability=0)
        # all_topics_csr = gensim.matutils.corpus2csc(self.topic_distributions)
        # all_topics_numpy = all_topics_csr.T.toarray()


    def get_corpus_from_id(self, idx):
        
        try:
            target_idx = self.doc_ids.index(idx)
            ret = self.corpuses[target_idx]
        except:
            ret = Result.NO_DOCS

        return ret

    def recommend_from_id(self, idx, num_similar_docs = 3):
        """
        TODO: 学習データのIDを指定すると、そのIDを当然推薦してくるので、それを除くこと。
        """ 
        
        if not self.is_model_trained:
            return Result.NO_MODEL# -2 # TODO: Define error codes

        # get topic distribution 
        test_corpus = self.get_corpus_from_id(idx)
        if test_corpus == Result.NO_DOCS:
            print("Doc does not exist")
            return Result.NO_DOCS# -1

        topic_dist = self.calc_topic_distribution_from_corpus(test_corpus)

        # get similarity from all training corpus
        similar_corpus_id = self.doc_index_similarity[topic_dist]

        # sort ids by similarity
        similar_corpus_id = sorted(enumerate(similar_corpus_id), key=lambda t: t[1], reverse=True) 
        
        if num_similar_docs > self.num_docs:
            num_similar_docs = self.num_docs
            print("[Warn] num similar docs must be less than self.num_docs")
        arr_ids = [ e[0] for e in similar_corpus_id[:num_similar_docs]]
        recommended_docs_ids = [self.doc_ids[e] for e in arr_ids]

        return recommended_docs_ids

    def calc_best_topic_from_id(self, idx):
        if not self.is_model_trained:
            return Result.NO_MODEL# -2 # TODO: Define error codes

        # get corpus
        test_corpus = self.get_corpus_from_id(idx)
        if test_corpus == -1:
            print("Doc does not exist")
            return Result.NO_DOCS# -1

        # get topic distribution
        topic_dist = self.calc_topic_distribution_from_corpus(test_corpus)
        ret = topic_dist[0][0]
        return ret


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
        topic_model.load_nltk_data(should_download=False)
        topic_model.set_num_topics(3)
        
        # topic_model.create_corpus_from_df(df)
        topic_model.create_corpus_from_test_data()

        topic_model.train()
        topic_model.calc_coherence()

        print(topic_model.lda.get_document_topics(common_corpus[0]))


        with open("./topic_model.pickle", "wb") as f:
             pickle.dump(topic_model, f)
        exit(1)
    with open("./topic_model.pickle", "rb") as f:
        
        topic_model = pickle.load(f)
        topic_model.load_nltk_data() # TODO: if use pickle, nltk_data dir is not set...

        print(topic_model.get_model_info())

        # show topics
        # print("Show topics ==================================================")
        # topic_model.vizualize_result()
        
        # dump topic words
        # print("Dump topic words =============================================")
        # topic_id = 1
        # topic_model.disp_topic_words(topic_id)    
        
        # recommend docs
        print("Recommend docs ===============================================")
        
        idx = 12000
        doc  = df.iloc[idx]["abstract"]
        
        topic_model.disp_topic_distribution(doc)
        topic_model.add_doc(doc, idx=idx)

        idx = 12000
        print("----")
        recommended_ids = topic_model.recommend_from_id(idx, num_similar_docs=10)
        if recommended_ids == Result.NO_MODEL or recommended_ids == Result.NO_DOCS:
            print("error")
            exit(-1)

        for ridx in recommended_ids:
            # print(topic_model.get_doc(ridx))
            print("[recommend ]" + str(ridx))


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
        topic_model.save_lda_vis_as_html(method="tsne")

        import webbrowser
        uri = 'file://' + '/Users/MiniBell/workspace/sazanami/app/pyldavis_output.html'
        webbrowser.open_new_tab(uri)

    print("Done")

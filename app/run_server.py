import sys
import os
import time

import pickle
import flask
import numpy as np
import pandas as pd
from redis import Redis

from topic_model import TopicModel
from error_definition import Result


# initialize our Flask application and pre-trained model
app = flask.Flask(__name__)
redis = Redis(host='redis', port=6379)

# Global
topic_model = None
df = pd.read_csv("./arxivs_data.csv")

@app.route('/')
def hello():
    redis.incr('hits') # TODO: global変数じゃなくて大丈夫？
    return 'Hello World! I have been seen %s times.' % redis.get('hits')

@app.route("/model/init", methods=["GET"])
def init_model():
    """
    TODO: should remove when release
    """
    global topic_model
    global df
    
    response = {
        "Content-Type": "application/json",
        "status_code": 999
    }    

    # read test data
    topic_model = TopicModel()
    topic_model.load_nltk_data()
    topic_model.set_num_topics(5) # TODO: shoud remove or set num topics with another way

    # for debug
    topic_model.create_corpus_from_df(df)
    
    response["status_code"] = 200
    
    return flask.jsonify(response)

@app.route("/model/save", methods=["GET"])
def save_model():
    """
    for better debug
    """
    global topic_model

    response = {
        "Content-Type": "application/json",
        "status_code": 999
    }    

    with open("./topic_model.pickle", "wb") as f:
        pickle.dump(topic_model, f)
        response["status_code"] = 200

    print("Save topic model as pickle")
    return flask.jsonify(response)

@app.route("/model/load", methods=["GET"])
def load_model():
    """
    for better debug
    """
    global topic_model

    response = {
        "Content-Type": "application/json",
        "status_code": 999
    }    

    topic_model = None
    with open("./topic_model.pickle", "rb") as f:        
        topic_model = pickle.load(f)
        topic_model.load_nltk_data() # TODO: if use pickle, nltk_data dir is not set...
        topic_model.set_topic_distribution_index() # TODO: consider where this function should be called
        response["status_code"] = 200

    print("Load topic model from pickle")

    return flask.jsonify(response)

@app.route("/model/train", methods=["POST"])
def model_train():
    """
    API POST /model/train
    TODO: arguments
    """
    global topic_model
    
    response = {
        "Content-Type": "application/json",
        "status_code": 999
    }    

    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "POST":

        # default params
        num_pass = 1
        num_topics = 5

        # get params
        if flask.request.get_json().get("num_pass"):
            num_pass = flask.request.get_json().get("num_pass")

        if flask.request.get_json().get("num_topics"):
            num_topics = flask.request.get_json().get("num_topics")

        # num topic
        topic_model.set_num_topics(num_topics)

        # train model
        ret = topic_model.train(num_pass=num_pass)

        # error handling
        if ret == Result.NO_CORPUS:
            response["status_code"] = 500
            response["error"] = "There is no corpus"
        
        elif ret == Result.NO_DICTIONARY:
            response["status_code"] = 500
            response["error"] = "Dictionary must be set"

        elif ret == Result.NO_NUM_TOPICS: # never, but just in case
            response["status_code"] = 500
            response["error"] = "Number of topics must be set"
            
        elif ret == Result.SUCCESS:
            response["status_code"] = 200

            # Save pickle
            with open("./topic_model.pickle", "wb") as f:
                pickle.dump(topic_model, f)
                response["status_code"] = 200
        
    return flask.jsonify(response)


@app.route("/model", methods=["GET"])
def model_info():
    """
    API GET /model
    """
    global topic_model

    response = {
        "status_code" : 999,
        "Content-Type": "application/json"
    }    

    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "GET":

        params = topic_model.get_model_info()
        response["num_topics"] = params["num_topics"]# num_topics
        response["num_docs"] = params["num_docs"] # num_docs    
        
        if params["date"] is None: # not trained yet
            response["status_code"] = 404 # Bad
            response["error"] = "Topic model has not been created"
        else:
            response["status_code"] = 200 # Good
            response["model_create_datetime"] = params["date"].strftime('%Y/%m/%d_%H:%M:%S')

    return flask.jsonify(response)


@app.route("/docs/<int:ind>", methods=["GET"])
def recommend(ind=None):
    """
    API GET /docs/:ind?num_docs=XX
    """
    global topic_model

    response = {
        "Content-Type": "application/json",
        "status_code": 999
    }    

    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "GET":
        num_similar_docs = flask.request.args.get("num_docs", 3)
        ret = topic_model.recommend_from_id_(ind, num_similar_docs = int(num_similar_docs))

        if ret == Result.NO_DOCS:
            response["status_code"] = 404
            response["error"] = "Document is not found"
            return flask.jsonify(response)

        if ret == Result.NO_MODEL:
            response["status_code"] = 500
            response["error"] = "Topic model is not created"
            return flask.jsonify(response)
        
        # TODO: 500: Something went wrong
        
        topic_no = topic_model.calc_best_topic_from_id(ind)
        response["topic"] = topic_no
        response["similar_docs"] = ret    
        response["status_code"] = 200

    return flask.jsonify(response)

@app.route("/docs/add", methods=["POST"])
def add_docs():
    """
    API
    """
    global topic_model
    
    response = {
        "Content-Type": "application/json",
        "status_code" : 999
    }

    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.get_json().get("doc_ind"):
            
            # read feature from json
            doc_ind = flask.request.get_json().get("doc_ind")

            # TODO: shoud remove because of DEBUG
            start = time.time()
            doc  = df.iloc[doc_ind]["abstract"] # TODO: handle out of index            
            print(time.time() - start, end="[sec]\n")

            start = time.time()
            ret = topic_model.add_doc(doc, ind=doc_ind)
            print(time.time() - start, end="[sec]\n")
            
            if ret == Result.SUCCESS:
                response["status_code"] = 200                
                # topic_model.set_topic_distribution_index() # calc topic distoributions with all corpus again TODO
            elif ret == Result.SAME_DOC:
                response["error"] = "The doc_id has already been used"
                response["status_code"] = 400
            else: # never, but just in case
                response["error"] = "Something went wrong"
                response["status_code"] = 500

    return flask.jsonify(response)

if __name__ == "__main__":

    print(" * Flask starting server...")
    app.run()

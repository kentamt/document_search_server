import sys
import os
import time
import glob
import pickle
from collections import deque

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
topic_model = TopicModel()
df = pd.read_csv("./arxivs_data.csv") # read DB here

pickles = sorted(glob.glob("./topic_model_*.pickle"))
if len(pickles) != 0: # Read Pickle
    latest_pickle = pickles[-1]
    print("[INFO ] Found pickle! Latest pickle is " + latest_pickle)

    with open(latest_pickle, "rb") as f:
        topic_model = pickle.load(f)
        topic_model.load_nltk_data() # TODO: if use pickle, nltk_data dir is not set...
        topic_model.set_topic_distribution_index() # TODO: consider where this function should be called
        print("[INFO ]Load topic model from " + latest_pickle)
else:
    # init model and data
    topic_model = TopicModel()
    topic_model.load_nltk_data()
    topic_model.set_num_topics(5) # TODO: shoud remove or set num topics with another way

    # read data from df
    topic_model.create_corpus_from_df(df)        

# --------------------------


@app.errorhandler(404)
@app.errorhandler(400)
@app.errorhandler(500)
def error_handler(error):
    '''
     Description
      - abort handler
    '''
    response = flask.jsonify(
    {
        "error": error.description['error']
    })

    return response, error.code

@app.route('/')
def hello():
    return "Hello world", 200

@app.route("/model/init", methods=["GET"])
def init_model():
    """
    TODO: should remove when release
    """
    global topic_model
    global df
    
    response = {}    

    try:
        # init model and data
        topic_model = TopicModel()
        topic_model.load_nltk_data()
        topic_model.set_num_topics(5) # TODO: shoud remove or set num topics with another way

        # for debug
        topic_model.create_corpus_from_df(df)        
        return flask.jsonify(response)
    except:
        flask.abort(500, {"error" : "Something went wrong."})


@app.route("/model/save", methods=["GET"])
def save_model():
    """
    for better debug
    """
    global topic_model

    if topic_model is None:
        flask.abort(404, {"error" : "Topic model has not been created."})

    response = {
        "Content-Type": "application/json",
        "status_code": 999
    }    

    params = topic_model.get_model_info()
    strtime = params["date"].strftime('%Y.%m.%d_%H.%M.%S')
    try:
        with open("./topic_model_" + strtime + ".pickle", "wb") as f:
            pickle.dump(topic_model, f)
            
            # delete old file if there are more than 10 files
            pickles = sorted(glob.glob("./topic_model_*.pickle"))
            if len(pickles) > 10:
                oldest_pickle = pickles[0]
                os.remove(oldest_pickle)
                print("Remove old pickle, " + oldest_pickle)

            response["status_code"] = 200
            
        print("Save topic model as pickle")
        return flask.jsonify(response)
    except:
        flask.abort(500, {"error" : "Something went wrong."})


@app.route("/model/load", methods=["GET"])
def load_model():
    """
    for better debug
    """
    global topic_model

    if topic_model is None:
        flask.abort(404, {"error" : "Topic model has not been created."})


    response = {}    
    try:
        topic_model = None

        pickles = sorted(glob.glob("./topic_model_*.pickle"))
        latest_pickle = pickles[-1]

        with open(latest_pickle, "rb") as f:        
            topic_model = pickle.load(f)
            topic_model.load_nltk_data() # TODO: if use pickle, nltk_data dir is not set...
            topic_model.set_topic_distribution_index() # TODO: consider where this function should be called
            print("Load topic model from pickle")

        return flask.jsonify(response)

    except:
        flask.abort(500, {"error" : "Something went wrong."})

@app.route("/model/train", methods=["POST"])
def model_train():
    """
    API POST /model/train
    TODO: arguments
    """
    global topic_model
    if topic_model is None:
        flask.abort(404, {"error" : "Topic model has not been created."})

    response = {}

    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "POST":

        # default params
        num_pass = 1
        num_topics = 5

        # get params
        try:
            if flask.request.get_json().get("num_pass"):
                num_pass = flask.request.get_json().get("num_pass")
        except:
            print("No json args")
        try:
            if flask.request.get_json().get("num_topics"):
                num_topics = flask.request.get_json().get("num_topics")
        except:
            print("No json args")
            
        # num topic
        topic_model.set_num_topics(num_topics)

        # train model
        ret = topic_model.train(num_pass=num_pass)

        # error handling
        if ret == Result.NO_CORPUS:
            print("here")
            flask.abort(500, {"error" : "There is no corpus"})
        
        elif ret == Result.NO_DICTIONARY:
            flask.abort(500, {"error" : "Dictionary must be set"})

        elif ret == Result.NO_NUM_TOPICS: # never, but just in case
            flask.abort(500, {"error" : "Number of topics must be set"})
            
        elif ret == Result.SUCCESS: # Save pickle
            
            params = topic_model.get_model_info()
            strtime = params["date"].strftime('%Y.%m.%d_%H.%M.%S')
            with open("./topic_model_" + strtime + ".pickle", "wb") as f:
                pickle.dump(topic_model, f)

                # delete old file if there are more than 10 files
                pickles = sorted(glob.glob("./topic_model_*.pickle"))
                if len(pickles) > 10:
                    oldest_pickle = pickles[0]
                    os.remove(oldest_pickle)
                    print("Remove old pickle, " + oldest_pickle)

        else: # just in case
            flask.abort(500, {"error" : "Something went wrong"})
        
    return flask.jsonify(response)


@app.route("/model", methods=["GET"])
def model_info():
    """
    API GET /model
    """
    global topic_model
    if topic_model is None:
        flask.abort(404, {"error" : "Topic model has not been created."})

    response = {}    

    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "GET":
  
        params = topic_model.get_model_info()
        response["num_topics"] = params["num_topics"]# num_topics
        response["num_docs"] = params["num_docs"] # num_docs    
        
        if params["date"] is None: # not trained yet
            flask.abort(404, {"error" : "Topic model has not been created."})
        else:
            response["model_create_datetime"] = params["date"].strftime('%Y/%m/%d_%H:%M:%S')

    return flask.jsonify(response)

@app.route("/docs/<int:idx>", methods=["GET"])
def recommend(idx=None):
    """
    API GET /docs/:idx?num_docs=XX
    """
    global topic_model

    if topic_model is None:
        flask.abort(404, {"error" : "Topic model has not been created."})

    response = {}    
    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "GET":
        num_similar_docs = flask.request.args.get("num_docs", 3)
        ret = topic_model.recommend_from_id(idx, num_similar_docs = int(num_similar_docs))

        if ret == Result.NO_DOCS:
            flask.abort(404, {"error" : "Document is not found."})

        if ret == Result.NO_MODEL:
            flask.abort(500, {"error" : "Topic model is not created."})
        
        topic_no = topic_model.calc_best_topic_from_id(idx)
        response["topic"] = topic_no
        response["similar_docs"] = ret    

    return flask.jsonify(response)

@app.route("/docs/add_idx", methods=["POST"])
def add_docs_idx():
    """
    API
    """
    global topic_model

    if topic_model is None:
        flask.abort(404, {"error" : "Topic model has not been created."})


    response = {}
    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.get_json().get("doc_idx"):
            
            # read feature from json
            doc_idx = flask.request.get_json().get("doc_idx")

            # TODO: shoud remove because of DEBUG
            start = time.time()
            
            try:
                doc  = df.iloc[doc_idx]["abstract"] # TODO: 0を入力するとエラーにならないが文献もaddされない．確認すること
            except:
                print("Out of Bounds or There is no data.")
                flask.abort(400, {"error" : "Invalid index."}) # TODO: I added new error to let user know this index is out of bound or invalid idx

            print(time.time() - start, end="[sec]\n")

            start = time.time()
            ret = topic_model.add_doc(doc, idx=doc_idx)
            print(time.time() - start, end="[sec]\n")
            
            if ret == Result.SUCCESS:
                # Save pickle
                params = topic_model.get_model_info()
                strtime = params["date"].strftime('%Y.%m.%d_%H.%M.%S')
                with open("./topic_model_" + strtime + ".pickle", "wb") as f:
                    pickle.dump(topic_model, f)

                    # delete old file if there are more than 10 files
                    pickles = sorted(glob.glob("./topic_model_*.pickle"))
                    if len(pickles) > 10:
                        oldest_pickle = pickles[0]
                        os.remove(oldest_pickle)
                        print("Remove old pickle, " + oldest_pickle)

            elif ret == Result.SAME_DOC:
                flask.abort(400, {"error" : "The document index has already been used."})

            else: # just in case
                flask.abort(500, {"error" : "Something went wrong."})
        else:
            flask.abort(500, {"error" : "Invalid parameters."})

    return flask.jsonify(response)

@app.route("/docs/add", methods=["POST"])
def add_docs():
    """
    API
    """
    global topic_model

    if topic_model is None:
        flask.abort(404, {"error" : "Topic model has not been created."})


    response = {}
    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.get_json().get("doc") and flask.request.get_json().get("idx"):
            
            # read document from json
            doc = flask.request.get_json().get("doc")
            doc_idx = flask.request.get_json().get("idx")
            ret = topic_model.add_doc(doc, idx=doc_idx)
            
            if ret == Result.SUCCESS:
                # Save pickle
                params = topic_model.get_model_info()
                strtime = params["date"].strftime('%Y.%m.%d_%H.%M.%S')
                with open("./topic_model_" + strtime + ".pickle", "wb") as f:
                    pickle.dump(topic_model, f)

                    # delete old file if there are more than 10 files
                    pickles = sorted(glob.glob("./topic_model_*.pickle"))
                    if len(pickles) > 10:
                        oldest_pickle = pickles[0]
                        os.remove(oldest_pickle)
                        print("Remove old pickle, " + oldest_pickle)

            elif ret == Result.SAME_DOC:
                flask.abort(400, {"error" : "The document index has already been used."})

            else: # just in case
                flask.abort(500, {"error" : "Something went wrong."})
        else:
            flask.abort(500, {"error" : "Invalid parameters."})

    return flask.jsonify(response)

if __name__ == "__main__":

    print(" * Flask starting server...")
    app.run()

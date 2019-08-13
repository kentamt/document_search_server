import pickle
import flask
import numpy as np
import pandas as pd
from topic_model import TopicModel
from redis import Redis

# initialize our Flask application and pre-trained model
app = flask.Flask(__name__)
redis = Redis(host='redis', port=6379)

# Global
topic_model = None

@app.route('/')
def hello():
    redis.incr('hits') # TODO: global変数じゃなくて大丈夫？
    return 'Hello World! I have been seen %s times.' % redis.get('hits')

@app.route("/model/init", methods=["GET"])
def init_model():
    global topic_model
    
    response = {
        "Content-Type": "application/json",
        "status_code": 999
    }    

    # read test data
    df = pd.read_csv("./arxivs_data.csv")
    topic_model = TopicModel()
    topic_model.load_nltk_data()
    topic_model.set_num_topics(5)
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
        ret = topic_model.train(num_pass=1)

        if ret == -1:
            response["status_code"] = 500
            response["error"] = "Something went wrong"
        else: # ret == 1:
            response["status_code"] = 200
            # response["perplexy"] = ...  TODO

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
            response["model_create_datetime"] = params["date"].strftime('%Y/%m/%d')

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
        ret = topic_model.recommend_from_id(ind, num_similar_docs = int(num_similar_docs))

        if ret == -1:
            response["status_code"] = 404
            response["error"] = "Document is not found"
            return flask.jsonify(response)

        if ret == -2:
            response["status_code"] = 500
            response["error"] = "Topic model is not created"
            return flask.jsonify(response)
        
        # TODO: 500: Something went wrong

        response["similar_docs"] = ret    
        
        topic_no = topic_model.calc_best_topic_from_id(ind)
        response["topic"] = topic_no

    return flask.jsonify(response)

@app.route("/docs/add", methods=["POST"])
def add_docs():
    """
    API
    """
    response = {
        "success": False,
        "Content-Type": "application/json"
    }

    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.get_json().get("doc_ind"):
            # read feature from json
            doc_ind = flask.request.get_json().get("doc_ind")
            print("rec doc ind: " + str(doc_ind))

    response["success"] = True
    return flask.jsonify(response)

if __name__ == "__main__":

    print(" * Flask starting server...")
    app.run()

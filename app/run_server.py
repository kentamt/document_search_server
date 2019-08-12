import pickle
import flask
import numpy as np
import pandas as pd
from topic_model import TopicModel

# initialize our Flask application and pre-trained model
app = flask.Flask(__name__)

# read test data
df = pd.read_csv("./arxivs_data.csv")
topic_model = TopicModel()
topic_model.load_nltk_data()
topic_model.set_num_topics(5)
topic_model.create_corpus_from_df(df)

#### USE PICKLE #####
# with open("./topic_model.pickle", "rb") as f:        
#     topic_model = pickle.load(f)
# topic_model.load_nltk_data()


@app.route("/model/train", methods=["POST"])
def model_train():
    """
    API
    TODO: arguments
    """

    response = {
        "success": False,
        "Content-Type": "application/json"
    }    

    print("Start training...")

    # ensure an feature was properly uploaded to our endpoint
    ret = -1
    if flask.request.method == "POST":
        ret = topic_model.train(num_pass=1)

    print("...End training")
    response["success"] = True

    return flask.jsonify(response)

@app.route("/model", methods=["GET"])
def model_info():
    """
    API
    """

    response = {
        "success": False,
        "Content-Type": "application/json"
    }    

    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "GET":
        num_topics, num_docs, model_create_datetime = topic_model.get_model_info()

    response["success"] = True
    response["num_topics"] = num_topics
    response["num_docs"] = num_docs
    response["model_create_datetime"] = model_create_datetime.strftime('%Y/%m/%d')

    return flask.jsonify(response)

@app.route("/docs/<int:ind>", methods=["GET"])
def recommend(ind=None):
    """
    API
    """
    response = {
        "success": False,
        "Content-Type": "application/json"
    }    

    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "GET":
        num_similar_docs = flask.request.args.get('num_similar', 3)
        ret = topic_model.recommend_from_id(ind, num_similar = int(num_similar_docs))

    if ret == -1:
        return flask.jsonify(response)

    response["success"] = True
    response["similar_docs"] = ret    

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
    # load_model()
    print(" * Flask starting server...")
    app.run()
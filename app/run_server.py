# built-in
import sys
import os
import signal
if os.name != 'nt':
    import fcntl
import time
import glob
import pickle
import atexit
import logging

import asyncio

# 3rd party
import flask
import pandas as pd

# local 
from topic_model import TopicModel
from error_definition import Result
import threading

# display training logs
logging.basicConfig(format='%(message)s', level=logging.INFO)

# initialize our Flask application and pre-trained model
app = flask.Flask(__name__)

# Initilize Topic Model
topic_model = TopicModel()

# Default value
FILE_NAME = "./data/test_data.csv"
CHUNK_SIZE = 50
NUM_MAX_DOCS = 400

# Read from env
FILE_NAME = os.environ['FILE_NAME']
CHUNK_SIZE = int(os.environ['CHUNK_SIZE'])
NUM_MAX_DOCS = int(os.environ['NUM_MAX_DOCS'])

logging.info("[INFO ] CSV file name : " +  FILE_NAME)
logging.info("[INFO ] Chunk size : " + str(CHUNK_SIZE))
logging.info("[INFO ] Max number of documents :  " + str(NUM_MAX_DOCS))


# loda model if there is model pickle
model_pickles = sorted(glob.glob("./model_*.pickle"))
if len(model_pickles) != 0:
    latest_model_pickle = model_pickles[-1]
    logging.info("[INFO ] Found model pickle! Latest pickle is " + latest_model_pickle)
    with open(latest_model_pickle, "rb") as f:
        if os.name != 'nt':
            fcntl.flock(f, fcntl.LOCK_EX)
        model = pickle.load(f)
        topic_model.set_model(model)
        logging.info("[INFO ] Load topic model from " + latest_model_pickle)
else:
    logging.info("[INFO ] There is no model pickles")
    
# load data if there is data pickle.
data_pickles = sorted(glob.glob("./data_*.pickle"))
if len(data_pickles) != 0:
    latest_data_pickle = data_pickles[-1]
    logging.info("[INFO ] Found data pickle! Latest pickle is " + latest_data_pickle)
    with open(latest_data_pickle, "rb") as f:
        if os.name != 'nt':
            fcntl.flock(f, fcntl.LOCK_EX)
        data = pickle.load(f)
        topic_model.set_data(data)
        topic_model.set_topic_distribution_index()
        logging.info("[INFO ] Load data from " + latest_data_pickle)

else:
    logging.info("[INFO ] Read data from csv")
    topic_model.create_corpus_from_csv(FILE_NAME, chunksize=CHUNK_SIZE, num_docs=NUM_MAX_DOCS)

def save_only_model():
    """ 
    save 10 latest model data as pickle
    """
    global topic_model
    
    # Save pickle
    params = topic_model.get_model_info()

    if params["date"] is None:
        logging.warning("[WARN ] No model to save.")
    else:
        strtime = params["date"].strftime('%Y.%m.%d_%H.%M.%S') 
        with open("./model_" + strtime + ".pickle", "wb") as f:
            if os.name != 'nt':
                fcntl.flock(f, fcntl.LOCK_EX)
            pickle.dump(topic_model.model, f)
            logging.info("[INFO ] Save model as pickle.")

        # delete old file if there are more than 10 files
        pickles = sorted(glob.glob("./model_*.pickle"))
        if len(pickles) > 10:
            oldest_pickle = pickles[0]
            os.remove(oldest_pickle)
            logging.info("[INFO ] Remove old model file, " + oldest_pickle)

def save_only_data():
    """
    save latest 10 corpus data as picke
    """
    global topic_model
    
    # Save pickle
    params = topic_model.get_model_info()

    if params["date"] is None:
        logging.warning("[WARN ] No data to save.")
    else:
        strtime = params["date"].strftime('%Y.%m.%d_%H.%M.%S') 
        with open("./data_" + strtime + ".pickle", "wb") as f:
            if os.name != 'nt':
                fcntl.flock(f, fcntl.LOCK_EX)
            pickle.dump(topic_model.data, f)
            logging.info("[INFO ] Save data as pickle.")

        # delete old file if there are more than 10 files
        pickles = sorted(glob.glob("./data_*.pickle"))
        if len(pickles) > 10:
            oldest_pickle = pickles[0]
            os.remove(oldest_pickle)
            logging.warning("[WARN ] Remove old data file, " + oldest_pickle)


is_train_running = False
class TrainThread(threading.Thread):
    """
    Threading class for multi thread training
    """
    def __init__(self, num_pass):
        super(TrainThread, self).__init__()
        self.stop_event = threading.Event()
        self.num_pass = num_pass

    def stop(self):
        self.stop_event.set()

    def run(self):
        global is_train_running, topic_model
        try:
            is_train_running = True
            topic_model.train(num_pass=self.num_pass)
            save_only_model()
        finally:
            is_train_running = False
            logging.info("[INFO ] Finish training.")

            
def signal_handler():
    """
    Save data as pickle before shutdown
    """
    logging.warning("[WARN ]signal handler is called!")
    # save_only_model()
    save_only_data()
    sys.exit(0)   

# set signal handler
atexit.register(signal_handler)

# ---------------------------------------------------
# Flask app
# ---------------------------------------------------
print("[INFO ] * Flask starting server...")

@app.errorhandler(404)
@app.errorhandler(400)
@app.errorhandler(500)
def error_handler(error):
    """
    abort handler
    """
    try:
        response = flask.jsonify(
            {
            "error": error.description['error']
            }
        )

        return response, error.code
    except: # for default 500 
        response = flask.jsonify(
        {
            "error": "Not found"
        }
    )
        return response, 404

@app.errorhandler(405)
def method_not_allowed(e):
    response = flask.jsonify(
        {
            "error": "Invalid method."
        }
    )
    return response, 405

@app.route("/model/train", methods=["POST"])
def model_train():
    """
    API POST /model/train
    """
    global topic_model, is_train_running
    
    if topic_model is None:
        flask.abort(404, {"error" : "Topic model has not been created."})

    # ensure method
    if flask.request.method == "POST":

        if is_train_running:
            logging.warn("[WARN ] traning is running.")
            flask.abort(500, {"error" : "Traning is running"})

        # default params
        num_pass = 5
        num_topics = 10

        # get params
        try:
            if flask.request.get_json().get("num_pass"):
                num_pass = flask.request.get_json().get("num_pass")
        except:
            logging.info("[INFO ] No json args")
        try:
            if flask.request.get_json().get("num_topics"):
                num_topics = flask.request.get_json().get("num_topics")
        except:
            logging.info("[INFO ] No json args")
            
        # num topic
        topic_model.set_num_topics(num_topics)

        # train model
        t = TrainThread(num_pass)
        t.start()

    response = {}
    response["message"] = "Success"
    return flask.jsonify(response)


@app.route("/model", methods=["GET"])
def model_info():
    """
    API GET /model
    """
    global topic_model
    if topic_model is None:
        logging.error("[ERROR ] Topic model is not created")
        flask.abort(404, {"error" : "Topic model is not created."})

    response = {}    

    # ensure method
    if flask.request.method == "GET":
  
        params = topic_model.get_model_info()
        response["num_topics"] = params["num_topics"]# num_topics
        response["num_docs"] = params["num_docs"] # num_docs    
        response["message"] = "Success"
        
        if params["date"] is None: # not trained yet
            logging.error("[ERROR ] Topic model is not created")
            flask.abort(404, {"error" : "Topic model is not created."})
        else:
            response["model_create_datetime"] = params["date"].strftime('%Y/%m/%d_%H:%M:%S')

    return flask.jsonify(response)


@app.route("/docs/<int:idx>", methods=["GET"])
def recommend(idx=None):
    """
    API GET /docs/:idx?num_similar=XX
    """
    global topic_model

    if topic_model is None:
        logging.error("[ERROR ] Topic model is not created")
        flask.abort(404, {"error" : "Topic model is not created."})

    response = {}    

    # ensure method
    if flask.request.method == "GET":

        filters = flask.request.args.getlist("filter[]")
        filters.append(idx) # add query itself
        filters = set(filters) # remove overlap
        filters = [int(e) for e in filters] 
        num_filters = len(filters)
        num_similar_docs = int(flask.request.args.get("num_similar", 3))

        # search
        ret = topic_model.recommend_from_id(idx, num_similar_docs = num_similar_docs + num_filters) # in case result filtered out

        if ret == Result.NO_DOCS:
            logging.error("[ERROR ] Document is not found")
            flask.abort(404, {"error" : "Document is not found."})

        if ret == Result.NO_MODEL:
            logging.error("[ERROR ] Topic model is not created")
            flask.abort(500, {"error" : "Topic model is not created."})

        # filter
        ret = [e for e in ret if not e in filters]

        # in case ret is longer than num_similar
        if len(ret) > num_similar_docs:
            ret = ret[:num_similar_docs]

        topic_no = topic_model.calc_best_topic_from_id(idx)
        response["topic"] = topic_no
        response["similar_docs"] = ret    
        response["message"] = "Success"

    return flask.jsonify(response)

@app.route("/docs/add", methods=["POST"])
def add_docs():
    """
    API POST /docs/add
    """
    global topic_model, is_train_running

    if is_train_running:
        logging.warn("[WARN ] traning is running.")
        flask.abort(500, {"error" : "Traning is running"})

    if topic_model is None:
        logging.error("[ERROR ] Topic model is not created")
        flask.abort(404, {"error" : "Topic model has not been created."})
    
    response = {
        "message" : "The doc is successfully added."
    }

    # ensure parameters were properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.get_json().get("doc_body") and flask.request.get_json().get("doc_id"):
            
            # read document from json
            doc_idx = flask.request.get_json().get("doc_id")
            doc = flask.request.get_json().get("doc_body")
            ret = topic_model.add_doc(doc, idx=doc_idx)
            
            if ret == Result.SUCCESS:
                logging.info("[INFO ] The doc is successfully added.")

            elif ret == Result.SAME_DOC:
                logging.error("[ERROR ] The doc_id has already been used")
                flask.abort(400, {"error" : "The doc_id has already been used."})

            elif ret == Result.NO_MODEL:
                logging.error("[ERROR ] Topic model is not created")
                flask.abort(500, {"error" : "Topic model is not created."})

            else: # just in case
                logging.error("[ERROR ] Something went wrong")
                flask.abort(500, {"error" : "Something went wrong."})

        else:
            logging.error("[ERROR ] Invalid parameters")
            flask.abort(500, {"error" : "Invalid parameters."})

    return flask.jsonify(response)

@app.route("/model/viz", methods=["GET"])
def get_viz_html():
     """
     API GET /model/viz
     Save result of LDA as HTML
     """
     topic_model.save_lda_vis_as_html(filename="./pyldavis_output.html", method="tsne")
     response = {
        "message" : "Success."
     }

     return flask.jsonify(response) # flask.current_app.send_static_file("./pyldavis_output.html") # 

if __name__ == "__main__":
    logging.error("[ERROR] use Flask." )
    # without uwsgi mode.
    # logging.info("[INFO ] * Flask starting server...")
    # app.run()
    # save_only_data()
    # save_only_model()
    # logging.info("[INFO ] End of the program.")
    

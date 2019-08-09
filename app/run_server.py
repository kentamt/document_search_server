import pickle
import flask
import numpy as np

from topic_model import TopicModel

# initialize our Flask application and pre-trained model
app = flask.Flask(__name__)
model = None


def load_model():
    global model
    print(" * Loading pre-trained model ...")
    
    model = None        
    with open("./topic_model.pickle", "rb") as f:
        model = pickle.load(f)
        model.load_nltk_data() # TODO: if use pickle, nltk_data dir is not set...
    print(' * Loading end')


@app.route("/predict", methods=["POST"])
def predict():
    response = {
        "success": False,
        "Content-Type": "application/json"
    }

    # ensure an feature was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.get_json().get("feature"):
            # read feature from json
            feature = flask.request.get_json().get("feature")

            # preprocess for classification
            # list  -> np.ndarray
            feature = np.array(feature).reshape((1, -1))

            doc = "across a far-reaching diversity of scientific and industrial applications, a general key problem involves relating the structure of time-series data to a meaningful outcome, such as detecting anomalous events from sensor recordings, or diagnosing patients from physiological time-series measurements like heart rate or brain activity. currently, researchers must devote considerable effort manually devising, or searching for, properties of their time series that are suitable for the particular analysis problem at hand. addressing this non-systematic and time-consuming procedure, here we introduce a new tool, hctsa, that selects interpretable and useful properties of time series automatically, by comparing implementations over 7700 time-series features drawn from diverse scientific literatures. using two exemplar biological applications, we show how hctsa allows researchers to leverage decades of time-series research to quantify and understand informative structure in their time-series data."
            response["query"] = doc

            # classify the input feature
            response["recommend"] = model.recommend(doc)[:5]

            # indicate that the request was a success
            response["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(response)


if __name__ == "__main__":
    load_model()
    print(" * Flask starting server...")
    app.run()
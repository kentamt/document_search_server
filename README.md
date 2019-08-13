# Sazanami

Document recommendation Web API tool.

- init_corpus(db_dir)
- set_parameters()
- train_model()
- update_mode()
- recommend(doc_dir)
- vizualize()
- add_docs(id or doc_dir) 
- optimize()

## Environment
- python 3.5.4 

## Build Docker Image
```
$ docker-compose build
```

## Run Docker on localhost

```
# listen on locahost:5000
$ docker-compose up
```

## Web API using Flask

For example:

```
$ python run_server.py
$ curl http://0.0.0.0:5000/predict -X POST -H 'Content-Type:application/json' -d '{"feature":[1, 1, 1, 1]}'
```

This returns json format like below.

```
{
  "Content-Type": "application/json", 
  "doc": "across a far-reaching diversity of scientific and industrial applications, a general key problem involves relating the structure of time-series data to a meaningful outcome, such as detecting anomalous events from sensor recordings, or diagnosing patients from physiological time-series measurements like heart rate or brain activity. currently, researchers must devote considerable effort manually devising, or searching for, properties of their time series that are suitable for the particular analysis problem at hand. addressing this non-systematic and time-consuming procedure, here we introduce a new tool, hctsa, that selects interpretable and useful properties of time series automatically, by comparing implementations over 7700 time-series features drawn from diverse scientific literatures. using two exemplar biological applications, we show how hctsa allows researchers to leverage decades of time-series research to quantify and understand informative structure in their time-series data.", 
  "recommend": [
    60, 
    35, 
    81
  ], 
  "success": true
}
```

## TODO
- Content-Type should be returned as response header?
- Status Code. sucess: 200, wrong endpoint: 404, internal error: 50X
- Add docker-compose and a container for database
- Hyper parameters optimization
- Save and load method
- Function for stopwords. i.e. add_stopwords(), show_stopwords()

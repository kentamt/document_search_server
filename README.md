# Sazanami

Sazanami is an old word that was always used with Shiga on Lake Biwa.

# Build Docker Image
```
$ docker build --tag=aaazzz/flask-uwsgi-nginx .
```

# Run Docker on localhost

```
# listen on locahost:80
$ docker run -p 80:80 aaazzz/flask-uwsgi-nginx

# listen on localhost:5000
$ docker run -p 5000:80 aaazzz/flask-uwsgi-nginx
```

# Web API using Flask

For example:

```
$ python run_server.py
$ curl http://0.0.0.0:5000/predict -X POST -H 'Content-Type:application/json' -d '{"feature":[1, 1, 1, 1]}'
```

This returns json format as below.

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

# TODO
- 
- 
- 
# Similar Document serch

Similar Document search web API tool.
![fish](./fish.jpg)

## Environment
- python 3.5.4 
- see requirements.txt  

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
$ curl http://0.0.0.0:5000/docs/10000?num_docs=3 -X GET
```

This returns json format like below.

```
{"doc_idx": [60, 35, 81], "topic_id": 1}
```

## TODO
- separeta train and recommend
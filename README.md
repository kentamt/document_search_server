# Similar Document search

Similar Document search web API tool.

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

run_server.py

```python
if __name__ == "__main__":
    # logging.error("[ERROR] use Flask." )

    # without uwsgi mode.
    app.run()
    save_only_data()
    save_only_model()
    logging.info("[INFO ] End of the program.")
```
then

```
$ python run_server.py
$ curl http://0.0.0.0:5000/docs/10000?num_docs=3 -X GET
```

This returns json format like below.

```
{"doc_idx": [60, 35, 81], "topic_id": 1}
```

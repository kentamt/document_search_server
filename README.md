# sazanami
document organizing tools development

# Build Docker Image
```
$ docker build --tag=aaazzz/flask-uwsgi-nginx .
```

# Run Docker on localhost

```
# listen on locahost:80
$ docker run -p 80:80 aaazzz/flask-uwsgi-nginx

# listen on localhost:5000
$ docker run -p 80:80 aaazzz/flask-uwsgi-nginx
```

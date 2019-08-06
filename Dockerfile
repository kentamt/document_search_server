FROM ubuntu:19.04
MAINTAINER aaazzz "akiramazvn@gmail.com"

RUN apt update && apt install -y nginx python-pip
RUN pip install --upgrade pip && pip install uwsgi flask supervisor

COPY . /app

WORKDIR /app
RUN mv etc/nginx.conf /etc/nginx/nginx.conf
RUN mv etc/uwsgi.ini /etc/uwsgi.ini
RUN mv etc/supervisord.conf /etc/supervisord.conf


EXPOSE 80
CMD ["/usr/local/bin/supervisord"]

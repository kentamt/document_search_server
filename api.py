from flask import Flask

app = Flask(__name__)

@app.route("/")
def greating():
    return "Welcome to Flask!"

if __name__ == '__main__':
    app.run(host="0.0.0.0", kkkkport=5000, debug=True)

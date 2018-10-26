
import time
import json
from flask import Flask
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple

app = Flask(__name__)

@app.route("/f24/test",methods=['POST'])
def f24Form():
    return "Hellow"


if __name__ == '__main__':
        #run_simple('localhost', 9000, app)
        app.run(debug=True, use_reloader=True)

	





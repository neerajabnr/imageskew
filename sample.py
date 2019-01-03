
import time
import json
import base64
import cv2
import numpy as np
from flask import Flask,request
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
import imgskew

app = Flask(__name__)

@app.route("/f24/api/sample",methods=['POST'])
def f24Form():
  print('started skewing images...')
  data = request.json
  #print(request.json)
  #print(data)
  #print(data['encoded_img'])
  #result = imgskew.imageSkew(data['encoded_img']) 
  #print(data['encoded_img'])
  #check(result)
  result=data['encoded_img']
  response = base64.b64decode(result)
  print(response
  return response

def check(imgbase64):
  nparr = np.fromstring(base64.b64decode(imgbase64), np.uint8)
  skewed_image = cv2.imdecode(nparr, 1)
  cv2.imwrite('./crossverify.jpg',skewed_image)


if __name__ == '__main__':
  #run_simple('localhost', 9000, app)
  app.run(debug=True, use_reloader=True)

	





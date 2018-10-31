
import time
import json
import base64
import cv2
import numpy as np
from flask import Flask,request
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
import imgskew
import gc

app = Flask(__name__)

@app.route("/f24/api/imageskew",methods=['POST'])
def f24Form():
  gc.collect()
  gc.garbage
  print('started skewing image...')
  data = request.json
  #print('request')
  #print(data)
  #print(data['encoded_img'])
  result = imgskew.imageSkew(data['encoded_img']) 
  #result="sample"
  #check(result)
  response = json.dumps({'encodedImage':result.decode('UTF-8')})
  gc.collect()
  gc.garbage
  return response

def check(imgbase64):
  nparr = np.fromstring(base64.b64decode(imgbase64), np.uint8)
  skewed_image = cv2.imdecode(nparr, 1)
  cv2.imwrite('./crossverify.jpg',skewed_image)


if __name__ == '__main__':
  #run_simple('localhost', 9000, app)
  app.run(debug=True, use_reloader=True)

	





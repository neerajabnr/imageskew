
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
import endpt
import match

app = Flask(__name__)



@app.route("/f24/api/imageskew",methods=['POST'])
def f24Form():
  print('started skewing images...')
  data = request.json
  #print(request.json)
  print(data)
  #
  print(data['encodedImage'])
  im_out = imgskew.imageSkew(data['encodedImage']) 
  #print(data['encodedImage'])
  #check(result)
  points = endpt.findpt(im_out)
 
  retval, buffer = cv2.imencode('.jpg', im_out)

    # Convert to base64 encoding and show start of data
  jpg_as_text = base64.b64encode(buffer)
    #print(jpg_as_text[:80])
  print(type(jpg_as_text))
  tickMarkCoords, tickMarkBoundingLimits = match.findTickCoordinates(im_out)
  tickMarkCoordsJSON = []
  for x in tickMarkCoords :
      pt  = {"topCorner":{"x":int(x[0][0]),"y":int(x[0][1])},"bottomCorner":{"x":int(x[1][0]),"y":int(x[1][1])}}
      tickMarkCoordsJSON.append(pt)
  response = json.dumps({'encodedImage':jpg_as_text.decode('UTF-8'),'bounds' : points,'tickMarkCoords' : tickMarkCoordsJSON , 'tickMarkBoundingLimits' :  tickMarkBoundingLimits})
  #print(response)
  return response

def check(imgbase64):
  nparr = np.fromstring(base64.b64decode(imgbase64), np.uint8)
  skewed_image = cv2.imdecode(nparr, 1)
  cv2.imwrite('./crossverify.jpg',skewed_image)



if __name__ == '__main__':
  run_simple('localhost', 4000, app)
  app.run(debug=True, use_reloader=True)





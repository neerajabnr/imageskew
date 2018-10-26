
# coding: utf-8

# In[1]:


from flask import Flask,request
from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
from PIL import Image
from io import BytesIO
import base64
import json

app_test = Flask(__name__)

@app.route("/test",methods=["POST"])
def hello():
    print(request.files)
    file = request.files['f24test']
    print(file)
    #file.save("f24.png")
    #with open(file.stream,'rb') as f:
    #  data = f.read()
    #print(type(file.read()))
    encodedVal = base64.b64encode(file.read())
    print(encodedVal)
    im = Image.open(BytesIO(base64.b64decode(encodedVal)))
    im.save('testt.png')
    return json.dumps({'encodedImg' : str(encodedVal) })

@app.route("/f24/imgencode",methods=["PUT"])
def imageEncode():
    #print(request.json)
    data = request.json
    print(data['imageb64'])
    imgData = bytes(data['imageb64'])
    decodedImg = base64.b64decode(imgData)
    print('--------------------------------------------')
    im = Image.open(BytesIO(decodedImg))
    im.save('testt12.png')
    return json.dumps({'status':'OK'})


if __name__ == '__main__':
    run_simple('localhost', 9000, app)


import numpy as np
import cv2
import base64
from matplotlib import pyplot as plt
import math
import gc
import imgmatch
#from PIL import Image

def imageSkew(imgbase64):
  #commenting the template code as we are using standard template
  pathofthetemplate=''
  #res=F24ImageFinder.findF24ImageType(imgbase64)
  #with open("imageToSave.png", "wb") as fh:
     #fh.write(imgbase64.decode('base64'))
  imgdata = base64.b64decode(imgbase64)
  filename = 'source_image.jpg'
  with open(filename, 'wb') as f:
    f.write(imgdata)

  pathofthetemplate=imgmatch.findTemplate('./source_image.jpg')
  #print(pathofthetemplate)
  orig_image = cv2.imread(pathofthetemplate, 0)
  #orig_image = cv2.imread('./F24_form.JPG', 0)
  #skewed_image = cv2.imread('./img_skew.jpg', 0)
  nparr = np.fromstring(base64.b64decode(imgbase64), np.uint8)
  #nparr = np.fromstring(base64.b64decode(encoded_string), np.uint8)
  skewed_image = cv2.imdecode(nparr, 1)
  #cv2.imwrite('./check.jpg',skewed_image)
  #skewed_image = cv2.imread(base64.encode(imgbase64),0)
  surf = cv2.xfeatures2d.SURF_create(400)
  kp1, des1 = surf.detectAndCompute(orig_image, None)
  kp2, des2 = surf.detectAndCompute(skewed_image, None)
  
  FLANN_INDEX_KDTREE = 0
  index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
  search_params = dict(checks=50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)
  matches = flann.knnMatch(des1, des2, k=2)
  
  # store all the good matches as per Lowe's ratio test.
  good = []
  for m, n in matches:
    if m.distance < 0.7 * n.distance:
      good.append(m)
  
  #print(good)
  
  MIN_MATCH_COUNT = 10
  if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
    ss = M[0, 1]
    sc = M[0, 0]
    scaleRecovered = math.sqrt(ss * ss + sc * sc)
    thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
    print("Calculated scale difference: %.2f\nCalculated rotation difference: %.2f" % (scaleRecovered, thetaRecovered))
    im_out = cv2.warpPerspective(skewed_image, np.linalg.inv(M), (orig_image.shape[1], orig_image.shape[0]))
    print("Calculated cv2.wrap")
    #plt.imshow(im_out, 'gray')
    cv2.imwrite('./sc.jpg',im_out)
    #im = Image.open('./sc.jpg')
    #width, height = im.size
    #print(height,width)
    #plt.show()
    # Convert captured image to JPG
    retval, buffer = cv2.imencode('.jpg', im_out)

    # Convert to base64 encoding and show start of data
    jpg_as_text = base64.b64encode(buffer)
    #print(jpg_as_text[:80])
    print(type(jpg_as_text))
    return jpg_as_text
    
  else:
    errorResult = "Not enough matches are found   -   %d/%d" % (len(good), MIN_MATCH_COUNT);
    print(errorResult)
    matchesMask = None
    return errorResult

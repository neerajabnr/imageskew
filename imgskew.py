import numpy as np
import cv2
import base64
from matplotlib import pyplot as plt
import math
import gc
import imgmatch

def imgDenoise(imgpath):
  img = cv2.imread(imgpath)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  img = clahe.apply(img)
  cv2.imwrite(imgpath,img)


def imageSkew(imgbase64):
  #commenting the template code as we are using standard template
  pathofthetemplate=''
  imgdata = base64.b64decode(imgbase64)
  filename = 'source_image.jpg'
  imgDenoise(filename)
  with open(filename, 'wb') as f:
    f.write(imgdata)

  pathofthetemplate=imgmatch.findTemplate('./source_image.jpg')
  orig_image = cv2.imread(pathofthetemplate, 0)
  skewed_image = cv2.imread(filename, 0)
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
    cv2.imwrite('./sc.jpg',im_out)
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

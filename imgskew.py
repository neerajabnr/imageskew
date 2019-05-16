import numpy as np
import cv2
import base64
from matplotlib import pyplot as plt
import math
import gc
import imgmatch

def imgDenoise(img):
  
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  img = clahe.apply(img)
  return img

def erosion(img):
  kernel = np.ones((2,2),np.uint8)
  img = cv2.erode(img,kernel,iterations = 1)
  return img

def adjustGamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def removelines(img_gray):


  
  template = cv2.imread('./patch/temp1.png',0)
  w, h = template.shape[::-1]
  res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
  threshold = 0.75
  loc = np.where( res >= threshold)
  img_cpy = img_gray
  for pt in zip(*loc[::-1]):
   
    p = ((pt[0]+10,pt[1]+4), (pt[0] + w-11, pt[1] + h-4)) 
    intensity = 255
    cv2.rectangle(img_cpy, p[0],p[1], intensity, cv2.FILLED)
    
  ret,mask = cv2.threshold(img_cpy,254,255,cv2.THRESH_BINARY)
  
  img_gray = cv2.inpaint(img_gray,mask,3,cv2.INPAINT_TELEA)
  return img_gray

def removeLinesInBetween(img_gray):


  
  template = cv2.imread('./patch/temp2.png',0)
  w, h = template.shape[::-1]
  res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
  threshold = 0.85
  loc = np.where( res >= threshold)
  img_cpy = img_gray
  for pt in zip(*loc[::-1]):
   
    p = ((pt[0]+3,pt[1]+3), (pt[0] + w-3, pt[1] + h-3)) 
    intensity = 255
    cv2.rectangle(img_cpy, p[0],p[1], intensity, cv2.FILLED)
    
  ret,mask = cv2.threshold(img_cpy,254,255,cv2.THRESH_BINARY)
  
  img_gray = cv2.inpaint(img_gray,mask,3,cv2.INPAINT_TELEA)
    


    
    
  return img_gray

   


def imageSkew(imgbase64):
  #commenting the template code as we are using standard template
  pathofthetemplate=''
  imgdata = base64.b64decode(imgbase64)
  filename = 'source_image.jpg'
  
  with open(filename, 'wb') as f:
    f.write(imgdata)

  #pathofthetemplate=imgmatch.findTemplate('./source_image.jpg')
  pathofthetemplate='./f24_templates/f24simp_2.jpg'
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
    im_out = imgDenoise(im_out)
    im_out = erosion(im_out)
    im_out = adjustGamma(im_out,1.5)
    im_out = removelines(im_out)
    #im_out = removeLinesInBetween(im_out)
    return im_out
    
  else:
    errorResult = "Not enough matches are found   -   %d/%d" % (len(good), MIN_MATCH_COUNT);
    print(errorResult)
    matchesMask = None
    return []



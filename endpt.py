from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse


def findEndPt(image ,temp,count):
  print(temp)
  img_object = cv.imread(temp, cv.IMREAD_GRAYSCALE)
  img_scene = cv.imread(image, cv.IMREAD_GRAYSCALE)
  if img_object is None or img_scene is None:
    print('Could not open or find the images!')
    exit(0)

#-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
  minHessian = 400
  detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
  keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
  keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)

#-- Step 2: Matching descriptor vectors with a FLANN based matcher
# Since SURF is a floating-point descriptor NORM_L2 is used
  matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
  knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)

#-- Filter matches using the Lowe's ratio test
  ratio_thresh = 0.7
  good_matches = []
  for m,n in knn_matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)
  print(len(good_matches))
  if len(good_matches)<5:
    return 0
#-- Draw matches
  img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
 
#-- Localize the object
  obj = np.empty((len(good_matches),2), dtype=np.float32)
  scene = np.empty((len(good_matches),2), dtype=np.float32)
  for i in range(len(good_matches)):
    #-- Get the keypoints from the good matches
    obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
    obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
    scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
    scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

  H, _ =  cv.findHomography(obj, scene, cv.RANSAC)

#-- Get the corners from the image_1 ( the object to be "detected" )
  obj_corners = np.empty((4,1,2), dtype=np.float32)
  obj_corners[0,0,0] = 0
  obj_corners[0,0,1] = 0
  obj_corners[1,0,0] = img_object.shape[1]
  obj_corners[1,0,1] = 0
  obj_corners[2,0,0] = img_object.shape[1]
  obj_corners[2,0,1] = img_object.shape[0]
  obj_corners[3,0,0] = 0
  obj_corners[3,0,1] = img_object.shape[0]

  scene_corners = cv.perspectiveTransform(obj_corners, H)

  if count<=3 :
    return(int((int(scene_corners[3,0,0])+int(scene_corners[0,0,0]))/2))
  else:
    return(int(scene_corners[2,0,0] ))

def findpt(img_path):
    
    points = []
    for i in range(1,10):
        img_template='./patch/'+str(i)+'.png';
        pt = findEndPt(img_path,img_template,i)
        points.append(pt)
    return points
if __name__ == '__main__':
       points = findpt('./sc.jpg')
       print(points)




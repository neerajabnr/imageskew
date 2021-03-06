#!/usr/bin/python
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sys
import getopt
from scipy import ndimage, misc
from math import sqrt
#import gi
#gi.require_version('Notify', '0.7')
#from gi.repository import Notify


INF = 999999999
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
SIMILARITY_TRESHOLD = 1.8         # Similarity value closer to zero means the images are more similar.
                                                        # Lower treshold means the check will be more strict (i.e. only images 
                                                        # that are very similar will be marked) and vice versa
TOP_MATCHES = 25
DISTANCE_TRESHOLD = 64
LOG_FLAG = False
BG_CHECK_TIMEOUT = 15
MESSAGE_TIMEOUT = 3
DAEMON_PID_FILE = '/tmp/imgmatch_daemon.pid'
DAEMON_LOCK_FILE = '/tmp/imgmatch_daemon.lock'

#
# Helper functions
#

def print_log(log, lv=1):
        if(LOG_FLAG):
                print (" "*lv)+">> "+str(log)

def print_help():
        print ("imgmatch is a tool to find duplicate images in a specified folder")
        print ("Usage example: python imgmatch.py -d <your_folder_path>")
        print ("Arguments: ")
        print ("  -h, --help          :  Show help and exit")
        print ("  -l, --log           :  Show log")
        print ("  -d, --dir=<DIR>     :  The directory to search for duplicate images")
        print ("                          default is current working directory")
        print ("  -r, --recursive     :  Search image recursively in subdirectories")
        print ("  -s <VALUE>          :  Similarity treshold to recognize duplicate images")
        print ("                          default value is 1.8, lower treshold means the")
        print ("                          check will be more strict (i.e. only images")
        print ("                          that are very similar will be marked) and vice versa")
        print ("  -b <start or stop>  :  Start or stop background mode")
        print ("                          when using -b option, imgmatch will run in ")
        print ("                          background watching a specified directory by ")
        print ("                          -d option. When duplicate image is found, ")
        print ("                          it will make a desktop notification.")

def print_err(err_code):
        if err_code == 1:
                print ("No/wrong argument. Use -h or --help for help")
        elif err_code == 2:
                print ("Error accessing specified folder")
        elif err_code == 3:
                print ("Error reading image files in specified folder")
        elif err_code == 4:
                print ("Background service currently not running (PID file not found)")
        else:
                print ("Unknown Error")

#
# Image processing functions
#

def find_matches(img1, img2):
        surf = cv2.xfeatures2d.SURF_create(400)
        kp1, des1 = surf.detectAndCompute(img1, None)
        kp2, des2 = surf.detectAndCompute(img2, None)
        index_params = dict()
        search_params = dict()
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        
# Apply ratio test
        good = []
        for m, n in matches:
          if m.distance < 0.7 * n.distance:
           good.append(m)
        #src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        #dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
        #M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
        #matchesMask = mask.ravel().tolist()
        #count = 0
        #for m in matchesMask:
          #if m > 0:
           #count = count+1
           
        #print(count)
  #print(good)

        # Sort them in the order of their distance.
        #good = sorted(good, key = lambda x:x.distance)
        # Only select top TOP_MATHCES points (if available)
	#good[:min(TOP_MATCHES, len(good))]
       
        
        #print(100*len(good)/len(matches))
        return len(good)

def compute_similarity(matches):
        # If the matches are too few, it's not likely to be duplicate
        if(len(matches) < TOP_MATCHES): 
                return INF

        # Compute the norm of the distances
        norm_d = 0
        for m in matches:
                norm_d += m.distance**2

        return sqrt(norm_d)/TOP_MATCHES

def is_duplicate(img1, img2):
        similarity = find_matches(img1, img2)
        #cv2.imwrite('img1.jpg',img1)
        #similarity = mse(img1, img2)
        #print(similarity)
        print_log(str(similarity),2)
        # If similarity value is less than certain treshold, the image is marked as duplicate
        #return (similarity < SIMILARITY_TRESHOLD) 
        return similarity



#
# Main Function
#
def findTemplate(img_path):
    #img = cv2.imread('./xx1.jpg',cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    #image_resized = misc.imresize(image, (768, 1024))
    #cv2.imwrite('./resizedimage.jpg',image_resized)
    #img = cv2.imread('./resizedimage.jpg',cv2.IMREAD_GRAYSCALE)
    #img = img.copy()
    similarityValue=0
    pathofthetemplate=''
    for i in range(1,5):
        img_template='./f24_templates/f24simp_'+str(i)+'.jpg'
        template = cv2.imread(img_template,cv2.IMREAD_GRAYSCALE)
        similarity=is_duplicate(img,template)
        print(similarity)
      
        if similarityValue<similarity:
           similarityValue=similarity
           pathofthetemplate=img_template
     
    print(pathofthetemplate)
    return pathofthetemplate



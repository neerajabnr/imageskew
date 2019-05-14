import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import endpt



def takeSecond(elem):
    return elem[1][1]

def findTickMark(img_gray,template):
    

    
    w, h = template.shape[::-1]
    res = cv.matchTemplate(img_gray,template,cv.TM_CCOEFF_NORMED)
    threshold = 0.65
    loc = np.where( res >= threshold)
    points = []
    pointA=[]
    cpy = img_gray
    for pt in zip(*loc[::-1]):
   
        p = ((pt[0]+3,pt[1]+3), (pt[0] + w-3, pt[1] + h-3)) 
        #cv.rectangle(cpy, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
   
        points.append(p)
    if len(points) > 0 :
       points = sorted(points , key=lambda k: [k[0][1], k[0][0]])
       print(points)
       merged =[]
       merged.append(points[0])
       for pt in points:
           last = merged[-1]
    
           if pt[0][1]-last[0][1] > 3 :
              merged.append(pt)
           else :
              merged[-1]=pt

       print(merged)
       for pt in merged:
           cv.rectangle(img_gray, pt[0],pt[1], (0,0,255), 2)
       return merged
    else :
       print('Inside else')
       return []

def findTickCoordinates(image):
    template = cv.imread('./patch/tick.png',0)
    points = findTickMark(image,template)
    if(len(points))>0:
       bounds = endpt.findTickBoundaries(image)
       return points,bounds
    else :
       return [] ,[]

   
if __name__ == '__main__':

   img_gray = cv.imread('lineremoval_14.jpg',0)
   points, bounds = findTickCoordinates(img_gray)
   print(points)
   print(bounds)

   #cv.imwrite('./tick/21.png',img_gray)


   



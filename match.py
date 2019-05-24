import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import endpt





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
        cv.rectangle(cpy, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
   
        points.append(p)
    cv.imwrite('op.jpg',cpy)
    if len(points) > 0 :
       return points
       
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
       template = cv.imread('./patch/cross.png',0)
       points = findTickMark(image,template)
       if(len(points))>0:
          bounds = endpt.findTickBoundaries(image)
          return points,bounds       
       else : 
          return [],[] 


   
if __name__ == '__main__':

   img_gray = cv.imread('sc.jpg',0)
   points, bounds = findTickCoordinates(img_gray)
   #cv.imwrite('op.jpg',img_gray)
   print(points)
   print(bounds)

   #cv.imwrite('./tick/21.png',img_gray)


   



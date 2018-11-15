import cv2
import numpy as np
from matplotlib import pyplot as plt

#img = cv2.imread('./img_skew.jpg',0)
#img2 = img.copy()
#template = cv2.imread('./f24simp_1.png',0)
#w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

def findF24ImageType(img_path):
    prevSS=0.0
    path_of_the_img=''
    #for meth in methods:
    for i in range(1,4):
        img_template='./f24_templates/f24simp_'+str(i)+'.png';
        template = cv2.imread(img_template,0)
        w, h = template.shape[::-1]
        img = cv2.imread(img_path,0)
        img = img.copy()
        #method = eval(meth)
        # Apply template Matching
        res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print(min_val)
        print(max_val)
        print(min_loc)
        print(max_loc)
        if prevSS ==0:
           prevSS=max_val
           path_of_the_img=img_template
        elif prevSS<max_val:
           prevSS=max_val
           path_of_the_img=img_template
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        #if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        #top_left = min_loc
        #else:
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img,top_left, bottom_right, 255, 2)
     
    print(path_of_the_img)
    return path_of_the_img
#plt.show()

import cv2
import numpy as np
import glob
 
img_array = []

for filename in sorted(glob.glob('./test/*.jpg')):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
    print(filename)
 
 
out = cv2.VideoWriter('test_24.07%.avi',cv2.VideoWriter_fourcc(*'DIVX'), 7, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
    print(i)
out.release()
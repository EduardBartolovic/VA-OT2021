import cv2
import os

image_folder = '../Output/Zugphase'
pathOut = '../Output/Zugphase/ZugphaseOF.avi'

images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'XVID'), 30, (width,height)) 
 
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

#import cv2
#import numpy as np
#import glob#
#
#frameSize = (1080, 1920)#

#out = cv2.VideoWriter('tracking.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)

#for filename in glob.glob('Output/SORTE/*jpg'):
#    img = cv2.imread(filename)
#    out.write(img)
#
#out.release()
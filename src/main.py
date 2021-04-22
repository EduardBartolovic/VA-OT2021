import cv2
import numpy as np

import time
import sys
import os
import math

from OpticalFlow import  opticalFlow, calculateMeanColorInBB
from YoloDetection import detect_image


# loading all the class labels (objects)labels
labels = open("../cfg/coco.names").read().strip().split("\n")

# generating colors for each object for later plotting
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

#Minimum Confidence
confidenceThreshold = 0.3

#Non-maximum suppression threshold
nmsThreshold = 0.2

#Config for drawing
font_scale = 1
thickness = 2

#Locations
videoInput = '../videos/Brudermuehl.mp4'
outputLocationOF = '../Output/OF'





"""
draw a bounding box rectangle and label on the image
Label could include direction and magintude
"""
def draw_detections(image, boxes, confidences, class_ids, location, direction = None, magintude = None):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold) #NMS function in opencv to perform Non-maximum Suppression
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i] # extract the bounding box coordinates
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}" #Define Text
            if not direction is None:   #ignore if directions and magnitude are not definded
                 
                endX = int(x+100*np.sin( direction[i][0]))
                endy = int(y+100*np.cos( direction[i][0]))

                degree = direction[i][0] * 180 / np.pi
                cv2.arrowedLine(image, (x,y), (endX, endy), (0, 0, 255), 3, 8, 0, 0.1)

                text +=  f"Dir: {degree:.2f} Mag: {magintude[i][0]:.2f}"


            text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            box_coords = ((x, y - 5), (x + text_width + 2, y - text_height - 5))
            overlay = image.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED) # Box for Text         
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0) # add opacity (transparency to the box)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness) # now put the text (label: confidence %)      

    cv2.imwrite(location+"/Output_" + str(count) + ".jpg", image)





#+++++++++++++++++++++++
vidcap = cv2.VideoCapture(videoInput)
success,image = vidcap.read()
image_h, image_w = image.shape[:2]
print('Image height:', image_h, ' Image width:', image_w)

prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

count = 0
while success:
    print('Frame:',count)
    success,image = vidcap.read()
    #if count > 2000:
    boxes,confidences,class_ids = detect_image(image, count)
    #draw_detections(image,boxes,confidences,class_ids,'Output') #Draw Normal with Bounding Boxes
    prev_gray, image, magnitude, angle = opticalFlow(prev_gray, image)
    magnitudes, angles = calculateMeanColorInBB(boxes, magnitude, angle, image_w, image_h)
    draw_detections(image,boxes,confidences,class_ids,outputLocationOF, magnitudes, angles) #Draw OF with Bounding Boxes
    count += 1

cap.release()
cv.destroyAllWindows()

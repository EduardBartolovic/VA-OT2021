import cv2
import numpy as np

import time
import sys
import os
import math

from OpticalFlow import  opticalFlow, calculateMeanColorInBB
from YoloDetection import detect_image
from trackingSort import *


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
videoInput = '../videos/../videos/Brudermuehl.mp4'
outputLocationOF = '../Output/OF'
outputLocationYOLO = '../Output/Yolo'
outputLocationSORT = '../Output/SORT'

"""
draw a bounding box rectangle and label on the image
Label could include direction and magintude
"""
def draw_detections(location, image, boxes, class_ids , confidences = None , direction = None, magintude = None):

    for i in range(len(boxes)):

        color = [int(c) for c in colors[class_ids[i]]]

        if confidences is None: #Part for tracking
            x, y, w, h, tracking_id = boxes[i] # extract the bounding box coordinates
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            tracking_id = int(tracking_id)
            cv2.rectangle(image, (x,y), (w,h), color=color , thickness=thickness)
            text = f"{labels[class_ids[i]]} {tracking_id}"
            
        else:

            x, y, w, h = boxes[i] # extract the bounding box coordinates
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"

            if not direction is None:   #draw arrows if directions and magnitude are definded
                endX = int(x+100*np.sin( direction[i][0]))
                endY = int(y+100*np.cos( direction[i][0]))
                cv2.arrowedLine(image, (x,y), (endX, endY), (0, 0, 255), 3, 8, 0, 0.1)
                #degree = direction[i][0] * 180 / np.pi
                #text +=  f"Dir: {degree:.2f} Mag: {magintude[i][0]:.2f}"

        text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        box_coords = ((x, y - 5), (x + text_width + 2, y - text_height - 5))

        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED) # Box for Text         
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0) # add opacity (transparency to the box)
        cv2.putText(image, text, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)    

    cv2.imwrite(location+"/Output_" + str(count) + ".jpg", image)


#+++++++++++++++++++++++
if not os.path.exists(outputLocationOF):
    os.makedirs(outputLocationOF)



vidcap = cv2.VideoCapture(videoInput)
success,image = vidcap.read()
image_h, image_w = image.shape[:2]
print('Image height:', image_h, ' Image width:', image_w)

prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sort_tracker = Sort()# tracker -> Sort

allDetections = []
allConfidences = []
count = 0
while success:
    print('Frame:',count)
    success,image = vidcap.read()
    #if count > 2000:
    boxes,confidences,class_ids = detect_image(image)
    allDetections.append(boxes)
    allConfidences.append(confidences)

    draw_detections(outputLocationYOLO,image,boxes,class_ids,confidences) #Draw Normal with Bounding Boxes

    tracking_boxes = []
    for d,c in zip(boxes,confidences):
        tracking_boxes.append([d[0],d[1],d[0]+d[2],d[1]+d[3],c])

    track_bbs_ids = sort_tracker.update(np.array(tracking_boxes))

    draw_detections(outputLocationSORT,image, track_bbs_ids, class_ids)

    prev_gray, image, magnitude, angle, mask = opticalFlow(prev_gray, image)
    magnitudes, angles = calculateMeanColorInBB(boxes, magnitude, angle, image_w, image_h, mask)
    draw_detections(outputLocationOF,image,boxes,class_ids,confidences, angles, magnitudes) #Draw OF with Bounding Boxes

    count += 1
    if count > 5:
        break

cv2.destroyAllWindows()

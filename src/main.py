import cv2
import numpy as np

import time
import sys
import os
import math

from OpticalFlow import opticalFlow, calculateMeanColorInBB
from YoloDetection import detect_image_yolo
from MOGDetection import detect_image_mog
from trackingSort import *
from Detection import Detection

from deepSort.trackingDeepSort import *                                         
from deepSort import nn_matching                                    
                                                                                
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
videoLocation = '../videos/'
outputLocationOF = '../Output/OF'     
outputLocationYOLO = '../Output/Yolo'  
outputLocationSORT = '../Output/SORT'
outputLocationMOG = '../Output/MOG'

# Video settings

#videoFile = 'goodtrain.mp4'
#allowedClasses = [2]# allows classes in which we are interested person,bicycle,car,motorbike,aeroplane,bus,train,truck
#crop_img_y = 0
#crop_img_x = 0
#crop_img_h = 1
#crop_img_w = 0.5
#max_cosine_distance = 0.4
#nn_budget = None
#max_iou_distance = 0.2
#metric = nn_matching.NearestNeighborDistanceMetric(
#    "euclidean", max_cosine_distance, nn_budget)  # DeepSort parameter

#Video Zug1

#Video Brudermühl
videoFile = 'Brudermuehl.mp4'
allowedClasses = [2]# allows classes in which we are interested person,bicycle,car,motorbike,aeroplane,bus,train,truck
crop_img_y = 0.25
crop_img_x = 0
crop_img_h = 1
crop_img_w = 0.75
max_cosine_distance = 0.4
nn_budget = None
max_iou_distance = 0.2
metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", max_cosine_distance, nn_budget)  # DeepSort parameter

#Video Candid
videoFile = 'Candidtunnel.mp4'
allowedClasses = [2]# allows classes in which we are interested person,bicycle,car,motorbike,aeroplane,bus,train,truck
crop_img_y = 0
crop_img_x = 0
crop_img_h = 1
crop_img_w = 1
max_cosine_distance = 0.4
nn_budget = None
max_iou_distance = 0.2
metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", max_cosine_distance, nn_budget)  # DeepSort parameter

#Video Kanal
#videoFile = 'Kanal.mp4'
#allowedClasses = [0]#allows classes in which we are interested person,bicycle,car,motorbike,aeroplane,bus,train,truck
#crop_img_y = 0
#crop_img_x = 0
#crop_img_h = 1
#crop_img_w = 1
#max_cosine_distance = 0.4
#nn_budget = None
#max_iou_distance = 0.7
#metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)# DeepSort parameter

"""
draw a bounding box rectangle and label on the image
Label could include direction and magintude
"""
def draw_detections(location, image, detections, direction = None, magintude = None):

    for i in range(len(detections)):

        color = [int(c) for c in colors[detections[i].get_class()]]

        x, y, w, h = detections[i].get_tlwh() # extract the bounding box coordinates
        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)

        if detections[i].get_tracking_id() is None: #Part for tracking
            text = f"{labels[detections[i].get_class()]}: {detections[i].get_confidence():.2f}"
        else:
            text = f"{labels[int(detections[i].get_class())]}: {int(detections[i].get_tracking_id())}"

        if not direction is None:   #draw arrows if directions and magnitude are definded
            if magintude[i] > 1:    #Only draw arrow if box is moving      
                endX = int(x+100*np.sin( direction[i][0]))
                endY = int(y+100*np.cos( direction[i][0]))
                cv2.arrowedLine(image, (x,y), (endX, endY), (0, 0, 255), 3, 8, 0, 0.1)
            #degree = direction[i][0] * 180 / np.pi
            #text +=  f"Dir: {degree} Mag: {magintude[i][0]:.1f}"

        text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        box_coords = ((x, y - 5), (x + text_width + 2, y - text_height - 5))

        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED) # Box for Text         
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0) # add opacity (transparency to the box)
        cv2.putText(image, text, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)    

    cv2.imwrite(location+"/Output_" + str(count) + ".jpg", image)




#+++++++++++++++++++++++



vidcap = cv2.VideoCapture(videoLocation+videoFile)
success,image = vidcap.read()

image_h, image_w = image.shape[:2]
print('Image height:', image_h, ' Image width:', image_w)
crop_img_y = int(crop_img_y*image_h)
crop_img_x = int(crop_img_x*image_w)
crop_img_h = int(crop_img_h*image_h)
crop_img_w = int(crop_img_w*image_w)
print('Cropped to Image:', crop_img_y,' + ', crop_img_h, ' + ', crop_img_x,' + ', crop_img_w)

#Crop image:
image = image[crop_img_y:crop_img_h, crop_img_x:crop_img_w]
image_h, image_w = image.shape[:2]

prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mog_object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# initialize tracker    
#sort_tracker = Sort()# tracker -> Sort                                                        
tracker = Tracker(metric,max_iou_distance) 

# initialize database   
#dataBase = DataBase()

count = 0
while success:

    print('Frame:',count)
    start = time.perf_counter()

    success,image = vidcap.read()

    image = image[crop_img_y:crop_img_h, crop_img_x:crop_img_w] #Crop image


    if count >= 0:

        ############################

        #Detection Start
        detections = detect_image_yolo(image,allowedClasses)

        
        draw_detections(outputLocationMOG, image, detect_image_mog(image, mog_object_detector))

        #draw_detections(outputLocationYOLO,image,detections)

        #Tracking SORT Start
        #tracking_boxes = [] 
        #for d in detections:
        #    t,l,b,r = d.get_tlbr()
        #    tracking_boxes.append([t,l,b,r,d.get_confidence()])
        #track_bbs_ids = sort_tracker.update(np.array(tracking_boxes))
        #trackingDetections = []
        #for d in track_bbs_ids:
        #    trackingDetections.append(Detection([d[0],d[1],d[2]-d[0],d[3]-d[1]], 0.0, 0, None, d[4]))
        #draw_detections(outputLocationSORT,image, trackingDetections)

        #Tracking DEEPSORT Start
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        track_bbs_ids = []                                                          
        for track in tracker.tracks:                                                
            if not track.is_confirmed() or track.time_since_update > 1:             
                continue                                                            
            bbox = track.to_tlbr()
            class_id = track.get_class()                                                  
            track_bbs_ids.append(Detection([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]], 0.0, class_id, None, track.track_id))
        draw_detections(outputLocationSORT,image, track_bbs_ids)  


        #OpticalFlow Start
        prev_gray, imageOF, magnitude, angle, mask = opticalFlow(prev_gray, image)
        magnitudes, angles = calculateMeanColorInBB(detections, magnitude, angle, image_w, image_h, mask)
        draw_detections(outputLocationOF, imageOF, detections, angles, magnitudes)

    count += 1

    time_took = time.perf_counter() - start
    print(f"Time took: {time_took:.2f}s")

    if count > 2000:
        break

cv2.destroyAllWindows()
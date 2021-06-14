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
from zugphase import zugphase

from deepSort.trackingDeepSort import *                                         
from deepSort import nn_matching        

from CounterBox import *
                                                                                
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
outputLocationDeepSORT = '../Output/DeepSORT'
outputLocationMOG = '../Output/MOG'
outputLocationSORT = '../Output/SORT'
outputLocationZugphase = '../Output/Zugphase'
#Classic or DeepLearning
classic = True


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
#videoFile = 'Candidtunnel.mp4'
#allowedClasses = [2,5]# allows classes in which we are interested person,bicycle,car,motorbike,aeroplane,bus,train,truck
#crop_img_y = 0.1
#crop_img_x = 0.1
#crop_img_h = 1
#crop_img_w = 1
#max_cosine_distance = 0.4
#nn_budget = None
#max_iou_distance = 0.2
#metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)# DeepSort parameter

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
def draw_detections(location, image, detections, show_state):

    for i in range(len(detections)):

        color = [int(c) for c in colors[detections[i].get_class()]]

        x, y, w, h = detections[i].get_tlwh() # extract the bounding box coordinates
        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)

        if detections[i].get_tracking_id() is None: #Part for tracking
            text = f"{labels[detections[i].get_class()]}: {detections[i].get_confidence():.2f}"
        elif show_state:
            id = detections[i].get_tracking_id()
            statestr = "kein Zug"
            if id in tracking_objects_states:
                state = tracking_objects_states[id].get_state()
                if state == 1:
                    statestr = "Einfahrend"
                elif state == 2:
                    statestr = "Haltend"
                elif state == 3:
                    statestr = "Abfahrend"
            text = f"{labels[int(detections[i].get_class())]}: {int(detections[i].get_tracking_id())}, {statestr}"
        else:
            text = f"{labels[int(detections[i].get_class())]}: {int(detections[i].get_tracking_id())}"

        #if not direction is None:   #draw arrows if directions and magnitude are definded
        if detections[i].dir_vec is not None:    #Only draw arrow if box is moving      
            endX = detections[i].dir_vec[0]
            endY = detections[i].dir_vec[1] 
            cv2.arrowedLine(image, (x,y), (x+int(endX*20), y+int(endY*20)), (0, 0, 255), 3, 8, 0, 0.1)
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


def classicPipeLine(image):

    #Detect Gaus:
    detections = detect_image_mog(image, mog_object_detector)
    draw_detections(outputLocationMOG, image, detections, False)

    #Tracking SORT:
    tracking_boxes = [] 
    for d in detections:
        t,l,b,r = d.get_tlbr()
        tracking_boxes.append([t,l,b,r, d.get_confidence()])

    trackingDetections = []
    if len(tracking_boxes) > 0:
        track_bbs_ids = tracker.update(np.array(tracking_boxes))
        for d in track_bbs_ids:
            trackingDetections.append(Detection([d[0], d[1], d[2]-d[0], d[3]-d[1]], 0.0, -1, None, d[4]))

    image = cv2.line(image, start_point, end_point, (255,255,255), 5) #Draw Line

    draw_detections(outputLocationSORT, image, trackingDetections, False)

    return trackingDetections

def deepLearningPipeLine(image):
    #Detect Yolo:
    detections = detect_image_yolo(image,allowedClasses)
    draw_detections(outputLocationYOLO, image, detections, False)

    #Tracking DEEPSORT:
    tracker.predict()
    tracker.update(detections)
    track_bbs_ids = []  
    angles = []
    magnitudes = []                                   
    for track in tracker.tracks:                                                
        if not track.is_confirmed() or track.time_since_update > 1:             
            continue       
                                                        
        bbox = track.to_tlbr()
        class_id = track.get_class()
        # Bestimmung der Richtung des Objekts
        mag, ang, dir_vec  = track.direction() 
        angles.append(ang)
        magnitudes.append(mag)                                      
        track_bbs_ids.append(Detection([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]], 0.0, class_id, None, track.track_id, dir_vec))

    image = cv2.line(image, start_point, end_point, (255,255,255), 5) #Draw Line

    draw_detections(outputLocationDeepSORT,image, track_bbs_ids, False)

    return track_bbs_ids, angles, magnitudes



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

# initialize tracker
if classic:
    mog_object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
    tracker = Sort()
else:                                                 
    tracker = Tracker(metric,max_iou_distance) #DeepSort

# initialize trackerBox   
counterBox = CounterBox(0*image_w, 0.7*image_h, 1*image_w, 0.9*image_h)#define TrackerBox tl,br
start_point = (0, int(0.7*image_h))
end_point = (image_w, int(0.7*image_h))

# Zugphase
tracking_objects_states = {}

count = 0
while success:

    print('Frame:',count)
    start = time.perf_counter()

    success,image = vidcap.read()

    if count >= 0:

        image = image[crop_img_y:crop_img_h, crop_img_x:crop_img_w] #Crop image
        

        angles = []
        magnitudes = []
        track_bbs_ids = []
        if classic:
            track_bbs_ids = classicPipeLine(image)
        else:
            track_bbs_ids, angles, magnitudes = deepLearningPipeLine(image)

        # Objektzähler
        counterBox.add( track_bbs_ids )
        #print(counterBox.getDict())

        # Zugphase
        #tracking_objects_states = zugphase(track_bbs_ids, angles, magnitudes, tracking_objects_states)
        #draw_detections(outputLocationZugphase,image, track_bbs_ids, True)
        

        #OpticalFlow:
        #prev_gray, imageOF, magnitude, angle, mask = opticalFlow(prev_gray, image)
        #magnitudes, angles = calculateMeanColorInBB(detections, magnitude, angle, image_w, image_h, mask)
        #draw_detections(outputLocationOF, image, detections, True, angles, magnitudes)

    count += 1

    time_took = time.perf_counter() - start
    print(f"Time took: {time_took:.2f}s")

    if count > 2000:
        break

cv2.destroyAllWindows()
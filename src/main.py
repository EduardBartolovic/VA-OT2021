import cv2
import numpy as np

import time
import sys
import os
import math

from OpticalFlow import  opticalFlow, calculateMeanColorInBB
from YoloDetection import detect_image
from trackingSort import *
from Detection import Detection

from deepSort.trackingDeepSort import *                                         
from deepSort import nn_matching                                                
#from deepSort.detection import Detection                                        
                                                                                
# loading all the class labels (objects)labels                                  
labels = open("/media/snow/HDD/Unizeug/VAOT/darknet/data/coco.names").read().strip().split("\n")
                                                                                
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
videoLocation = '/media/snow/HDD/Unizeug/VAOT/VA-OT2021/'
outputLocationOF = '/media/snow/HDD/Unizeug/VAOT/VA-OT2021/Output/OF'           
outputLocationYOLO = '/media/snow/HDD/Unizeug/VAOT/VA-OT2021/Output/Yolo'       
outputLocationSORT = '/media/snow/HDD/Unizeug/VAOT/VA-OT2021/Output/SORT'  


# loading all the class labels (objects)labels
#labels = open("../cfg/coco.names").read().strip().split("\n")

# generating colors for each object for later plotting
#colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

#Minimum Confidence
#confidenceThreshold = 0.3

#Non-maximum suppression threshold
#nmsThreshold = 0.2

#Config for drawing
#font_scale = 1
#thickness = 2

#Locations
#videoLocation = '../videos/'
#outputLocationOF = '../Output/OF'
#outputLocationYOLO = '../Output/Yolo'
#outputLocationSORT = '../Output/SORT'



#Video settings
#videoFile = 'Brudermuehl.mp4'
videoFile = 'videos/20210409_100728.mp4'
crop_img_y = 0.25
crop_img_x = 0
crop_img_h = 1
crop_img_w = 0.75

#videoFile = '1.mp4'
#crop_img_y = 0.50
#crop_img_x = 0.50
#crop_img_h = 1
#crop_img_w = 1

"""
draw a bounding box rectangle and label on the image
Label could include direction and magintude
"""
def draw_detections(location, image, detections, direction = None, magintude = None):

    for i in range(len(detections)):

        color = [int(c) for c in colors[detections[i].get_class()]]

        x, y, w, h = detections[i].get_tlwh() # extract the bounding box coordinates
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
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
if not os.path.exists(outputLocationOF):
    os.makedirs(outputLocationOF)



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

sort_tracker = Sort()# tracker -> Sort

max_cosine_distance = 0.4                                                       
nn_budget = None                                                                
# calculate cosine distance metric                                              
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# initialize tracker                                                            
tracker = Tracker(metric) 

count = 0
while success:

    print('Frame:',count)
    start = time.perf_counter()

    success,image = vidcap.read()

    image = image[crop_img_y:crop_img_h, crop_img_x:crop_img_w] #Crop image

    if True:#count > 2000:

        #Detection Start
        detections = detect_image(image)
        draw_detections(outputLocationYOLO,image,detections)

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
            track_bbs_ids.append(Detection([bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]], 0.0, 0, None, track.track_id))
        draw_detections(outputLocationSORT,image, track_bbs_ids)  

        #OpticalFlow Start
        prev_gray, imageOF, magnitude, angle, mask = opticalFlow(prev_gray, image)
        magnitudes, angles = calculateMeanColorInBB(detections, magnitude, angle, image_w, image_h, mask)
        draw_detections(outputLocationOF, imageOF, detections, angles, magnitudes)


    count += 1

    time_took = time.perf_counter() - start
    print(f"Time took: {time_took:.2f}s")

    if count > 5:
        break

cv2.destroyAllWindows()
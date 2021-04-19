import cv2
import numpy as np

import time
import sys
import os
import math


# the neural network configuration
config_path = "cfg/yolov4.cfg"

# the YOLO net weights file
weights_path = "weights/yolov4.weights" 

# loading all the class labels (objects)labels
labels = open("data/coco.names").read().strip().split("\n")

# generating colors for each object for later plotting
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

# load the YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

#net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#Minimum Confidence
confidenceThreshold = 0.3

#Non-maximum suppression threshold
nmsThreshold = 0.2

#Config for drawing
font_scale = 1
thickness = 2

#Locations



"""
return Bounding Boxes top left corner and height and width
"""
def detect_image(image,count):

    h, w = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)# create 4D blob

    #print("image.shape:", image.shape, "=> blob.shape:", blob.shape)

    net.setInput(blob) # sets the blob as the input of the network
    
    ln = net.getLayerNames() # get all the layer names
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # measure how much it took in seconds for the inference
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    print(f"Time took: {time_took:.2f}s")

    boxes, confidences, class_ids = [], [], []
    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            
            if class_id <= 8: #remove classes in which we are not interested person,bicycle,car,motorbike,aeroplane,bus,train,truck
                if confidence > confidenceThreshold: #remove weak predictions by ensuring the detected probability is greater than the threshold

                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes width and height
                    box = detection[:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    return boxes,confidences,class_ids

def opticalFlow(prev_gray, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Sets image hue according to the optical flow direction
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    return (gray, rgb, magnitude, angle)

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
                degree = direction[i][0] * 180 / np.pi / 2 
                endX = int(x+100*np.cos(direction[i][0]))
                endy = int(y+100*np.sin(direction[i][0]))
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



def calculateMeanColorInBB(boxes, magnitude, angles, image_w, image_h):
    count= 0
    magnitudeMeans = np.zeros((len(boxes),1))
    angleMeans = np.zeros((len(boxes),1))
    
    for box in boxes:

        startY = max(box[0],0)
        endY = min(box[0]+box[2], image_w-1)
        startX = max(box[1],0)
        endX = min(box[1]+box[3], image_h-1)
        #print('startX',startX,'endx',endX,'startY',startY,'endY',endY)

        mags = []
        angs = []
        for y in range(startY,endY):
            for x in range(startX,endX):
                mags.append(magnitude[x][y])
                angs.append(angle[x][y])
        
        magnitudeMeans[count] = np.mean(mags)
        angleMeans[count] = np.mean(angs)
        count += 1
    return (magnitudeMeans, angleMeans)



#+++++++++++++++++++++++
#vidcap = cv2.VideoCapture('videos/Brudermuehl.mp4')
vidcap = cv2.VideoCapture('videos/2.mp4')

success,image = vidcap.read()
image_h, image_w = image.shape[:2]
print('Image height:', image_h, ' Image width:', image_w)
prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(image)
mask[..., 1] = 255

count = 0
while success:
    print('Frame:',count)
    success,image = vidcap.read()
    if count > 2000:
        boxes,confidences,class_ids = detect_image(image, count)
        #draw_detections(image,boxes,confidences,class_ids,'Output') #Draw Normal with Bounding Boxes
        prev_gray, image, magnitude, angle = opticalFlow(prev_gray, image)
        magnitudes, angles = calculateMeanColorInBB(boxes, magnitude, angle, image_w, image_h)
        draw_detections(image,boxes,confidences,class_ids,'OutputOF', magnitudes, angles) #Draw OF with Bounding Boxes
    count += 1

cap.release()
cv.destroyAllWindows()

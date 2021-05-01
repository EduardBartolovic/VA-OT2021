import cv2
import numpy as np
import time
from Detection import Detection
from tools import generate_detections as gdet

# the neural network configuration                                              
#config_path = "/media/snow/HDD/Unizeug/VAOT/darknet/cfg/yolov4.cfg"             
config_path = "../cfg/yolo4.cfg"

# the YOLO net weights file                                                     
#weights_path = "/media/snow/HDD/Unizeug/VAOT/darknet/yolov4.weights"
weights_path = "../weights/yolov4.weights" 

# load the YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

#net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#Minimum Confidence
confidenceThreshold = 0.3

#Non-maximum suppression threshold
nmsThreshold = 0.2

# initialize deep sort
#model_filename = '/media/snow/HDD/Unizeug/VAOT/VA-OT2021/src/model_data/mars-small128.pb'
model_filename = '../weights/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

def choose_elements_by_indices(list_object, indices):
    newList = []
    for i in range(len(list_object)):
        if i in indices:
            newList.append(list_object[i])
    return newList


"""
return Bounding Boxes top left corner and height and width
"""
def detect_image(image):

    h, w = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)# create 4D blob

    #print("image.shape:", image.shape, "=> blob.shape:", blob.shape)

    net.setInput(blob) # sets the blob as the input of the network
    
    ln = net.getLayerNames() # get all the layer names
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # measure how much it took in seconds for the inference
    #start = time.perf_counter()
    layer_outputs = net.forward(ln)
    #time_took = time.perf_counter() - start
    #print(f"Time took: {time_took:.2f}s")

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

    #print('length',len(boxes))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold) #NMS function in opencv to perform Non-maximum Suppression
    boxes = choose_elements_by_indices(boxes, indices)
    confidences = choose_elements_by_indices(confidences, indices)
    class_ids = choose_elements_by_indices(class_ids, indices)
    #print('length',len(boxes))
    features = encoder(image, boxes)
    detections = [Detection(box, confidence, class_id, feature) for box, confidence, class_id, feature in zip(boxes, confidences, class_ids, features)]

    return detections #boxes,confidences,class_ids

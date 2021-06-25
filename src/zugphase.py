import numpy as np
from numpy.lib.function_base import diff
from Tracking_object import Tracking_object
import OpticalFlow as of
# States
# 0: Einfahren
# 1: Halten
# 2: Abfahren

# Richtungen:
# Threshold in der die Richtung von der aktuellen Richtung abweichen darf
# festgelegt auf 45 grad

# bboxes of objects
# direction of objects calculated through optical flow
# ids

def zugphase(detections, directions, magnitudes, tracking_objects_states, image_prev, image, useKalman):
    distance = 20
    mean_directions = 0
    last_directions = []
    for i, detection in enumerate(detections):
        if detection.get_class() == 6:
            id = detection.get_tracking_id()
            if id is not None and id in tracking_objects_states:
                tracking_object = tracking_objects_states[id]
                last_directions = tracking_object.get_last_directions()
                mean_directions = np.mean(last_directions)
                tracking_object.add_directions(directions[i])
                if not useKalman:
                # Use Optical Flow
                    magnitude = check_optical_flow(detection, image_prev, image)
                else:
                # Use Kalman Filter
                    magnitude = magnitudes[i]*10
                if magnitude > 1:
                    if directions[i] > (mean_directions - distance) and directions[i] < (mean_directions + distance):
                        tracking_object.increase_same_direction_counter()
                else:
                    tracking_object.increase_stop_counter()
            else:
                tracking_objects_states[id] = detection_to_tracker_object(detection)
    return tracking_objects_states

def detection_to_tracker_object(detection):
    return Tracking_object(detection.get_class(), detection.get_tracking_id())

def check_optical_flow(detection, image_prev, image):
    diff = 20
    image_h, image_w = image.shape[:2]
    x,y,_,_ = detection.to_xyah()
    startY = int(max(y-diff,0))
    endY = int(min(y+diff, image_w-1))
    startX = int(max(x-diff,0))
    endX = int(min(x+diff, image_h-1))
    image = image[startY:endY, startX:endX]
    image_prev = image_prev[startY:endY, startX:endX]
    magnitude,_, mask = of.opticalFlow(image_prev, image)
    res = of.is_moving(magnitude, mask)
    #print(res)
    return res
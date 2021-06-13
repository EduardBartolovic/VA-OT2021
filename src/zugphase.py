import numpy as np
from Tracking_object import Tracking_object
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

def zugphase(detections, directions, magnituedes, tracking_objects_states):
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
                if magnituedes[i] > 1:
                    if directions[i] > (mean_directions - distance) and directions[i] < (mean_directions + distance):
                        tracking_object.increase_same_direction_counter()

                else:
                    tracking_object.increase_stop_counter()
            else:
                tracking_objects_states[id] = detection_to_tracker_object(detection)
    return tracking_objects_states

def detection_to_tracker_object(detection):
    return Tracking_object(detection.get_class(), detection.get_tracking_id())
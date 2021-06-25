import cv2
import numpy as np

def opticalFlow(prev_gray, frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Sets image hue according to the optical flow direction
    mask = np.zeros_like(frame)
    mask[..., 1] = 255
    mask[..., 0] = angle * 180 / np.pi / 2
    # Sets image value according to the optical flow magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    return (magnitude, angle, mask)


def is_moving(magnitude, mask):
    mags = []
    for y in range(0,len(magnitude)):
        for x in range(0, len(magnitude)):
            if mask[x][y][2] > 19:
                mags.append(magnitude[x][y])
    mags.append(-1)
    magnitudeMean = np.mean(mags)
    print("magMean: ",magnitudeMean)
    return magnitudeMean
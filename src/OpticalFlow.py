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
    return (gray, rgb, magnitude, angle)


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
                if magnitude[x][y] > 0.1:
                    mags.append(magnitude[x][y])
                    angs.append(angles[x][y])

        mags.append(-1)
        angs.append(0)
        magnitudeMeans[count] = np.mean(mags)
        angleMeans[count] = np.mean(angs)

        count += 1
    return (magnitudeMeans, angleMeans)
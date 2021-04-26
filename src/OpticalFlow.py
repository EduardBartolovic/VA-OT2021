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
    return (gray, rgb, magnitude, angle, mask)


def calculateMeanColorInBB(detections, magnitude, angles, image_w, image_h, mask):
    count= 0
    magnitudeMeans = np.zeros((len(detections),1))
    angleMeans = np.zeros((len(detections),1))
    for det in detections:
        t,l,b,r = det.get_tlbr()
        startY = max(t,0)
        endY = min(b, image_w-1)
        startX = max(l,0)
        endX = min(r, image_h-1)

        mags = []
        angs = []
        col = []
        for y in range(startY,endY):
            for x in range(startX,endX):
                if mask[x][y][2] > 0:
                    col.append(mask[x][y][0])
                #if magnitude[x][y] > 0.1:
                mags.append(magnitude[x][y])
                angs.append(angles[x][y])

        mags.append(-1)
        angs.append(0)
        col.append(-1)
        magnitudeMeans[count] = np.mean(mags)
        #angleMeans[count] = np.mean(angs)
        meancol = np.mean(col)*2
        if (meancol > 90 and meancol < 180) or meancol > 270:
            angleMeans[count] = np.mean(angs)-np.pi
        else:
            angleMeans[count] = np.mean(angs)
        

        count += 1
    return (magnitudeMeans, angleMeans)
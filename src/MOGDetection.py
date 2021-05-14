import cv2
from Detection import Detection

def detect_image_mog(frame, object_detector):
    boxes = []
    #frame = cv2.resize(frame, (1280, 720))
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #roi = frame[0: 720, 0: 1280]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000: # Parameter
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append([x, y, int(w), int(h)])

    detections = [Detection(box, 100, 2, None) for box in boxes]
    return detections



#'''
#cap = cv2.VideoCapture("Brudermuehl.mp4")
#object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
#ret, frame1 = cap.read()
#while ret:
#    x1, y1, w1, h1 = detect_image(frame1, object_detector)
#    print(x1, y1, w1, h1)
#    ret, frame1 = cap.read()
#'''
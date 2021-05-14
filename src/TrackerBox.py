from Detection import Detection

class TrackerBox(object):

    def __init__(self,x,y,w,h):
        self.trackerBox = dict()
        self.x = x
        self.y = y
        self.w = w
        self.h = h


    def add(self, detections):
        for det in detections:
            _,y,_,h = det.get_tlbr()
            if y > self.y and y < self.h:
                if not det.get_tracking_id() in self.trackerBox:
                    print('New Object counted', (det.get_tracking_id(), det.get_class()) )
                    self.trackerBox[det.get_tracking_id()] = [det.get_class()]

    def getSize(self):
        return len(self.trackerBox)

    def getDict(self):
        return self.trackerBox
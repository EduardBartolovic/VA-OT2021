from Detection import Detection

class CounterBox(object):

    def __init__(self,x,y,w,h):
        self.counterBox = dict()
        self.x = x
        self.y = y
        self.w = w
        self.h = h


    def add(self, detections):
        for det in detections:
            _,y,_,h = det.get_tlbr()
            if y > self.y and y < self.h:
                if not det.get_tracking_id() in self.counterBox:
                    print('New Object counted', (det.get_tracking_id(), det.get_class()) )
                    self.counterBox[det.get_tracking_id()] = [det.get_class()]

    def getSize(self):
        return len(self.counterBox)

    def getDict(self):
        return self.counterBox
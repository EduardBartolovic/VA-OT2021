import cv2
import numpy as np
import time
import sys
import os
import math


class Database(object):

    def __init__(self):
        self.trackingHistory = dict()

	def add_boxes(self, detections):
        for det in detections:
            if det.get_tracking in trackingHistory:
                trackingHistory[det.get_tracking_id].append(det)
            else:
                trackingHistory[det.get_tracking_id] = [det]

    def getObjectHistory(self, trackid){
        return self.trackingHistory[trackid]

    def getLenOfObjectHistory(self, trackid){
        return len(self.trackingHistory[trackid])

    def getSize(self){
        return len(self.trackingHistory)

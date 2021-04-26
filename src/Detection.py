import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.
    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.
    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    class_name : ndarray
        Detector class.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.
    """

    def __init__(self, tlwh, confidence, class_name, feature, tracking_id=None):
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.class_name = class_name
        self.feature = np.asarray(feature, dtype=np.float32)
        self.tracking_id = tracking_id

    def get_class(self):
        return self.class_name

    def get_tracking_id(self):
        return self.tracking_id

    def get_confidence(self):
        return self.confidence

    def get_tlwh(self):
        return self.tlwh[0], self.tlwh[1], self.tlwh[2], self.tlwh[3]

    def get_tlbr(self):
        return self.tlwh[0], self.tlwh[1], self.tlwh[0]+self.tlwh[2], self.tlwh[1]+self.tlwh[3]

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
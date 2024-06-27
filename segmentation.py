#######################
# Object segmentation #
#######################

import supervision as sv
# Own modules
from detection import Detect
from settings import setting

class Segment(Detect):

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self):
        super().__init__()

        # Object for instance segmentation masks
        # https://roboflow.com/how-to-plot/yolov8-segmentation
        self.mask_annotator = sv.MaskAnnotator(
            color=sv.Color.BLUE
        )

    #############################################################################################################
    # CALL:

    def __call__(self, img):
        # Predict objects in image/frame
        results = self.predict(img)  
        # Read detections from image/frame
        detections = self.read_detections(results[0])
        # Draw bounding boxes with labels
        img = self.annotate_bboxes(img, detections)   
        # Draw segmentation masks
        img = self.mask_annotator.annotate(scene=img, detections=detections)  
        return img 
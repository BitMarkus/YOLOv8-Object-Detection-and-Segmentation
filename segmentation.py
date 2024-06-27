#######################
# Object segmentation #
#######################

from ultralytics.utils.plotting import Annotator, colors
import supervision as sv
# Own modules
from detection import Detect
from settings import setting

class Segment(Detect):

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self):
        super().__init__()

        # Outline thickness and color of segmentation masks
        self.show_show_mask = setting["os_show_mask"]  
        self.show_outline = setting["os_show_outline"]  
        self.outline_thickness = setting["os_outline_thickness"]  
        self.outline_color = setting["os_outline_color"]  

        # Object for instance segmentation masks
        # https://roboflow.com/how-to-plot/yolov8-segmentation
        self.mask_annotator = sv.MaskAnnotator(
            color=sv.Color.BLUE
        )

    #############################################################################################################
    # METHODS:

    # Draw outlines of segmented objects
    # https://docs.ultralytics.com/reference/utils/plotting/#ultralytics.utils.plotting.Annotator.seg_bbox
    def annotate_outlines(self, img, result):
        annotator = Annotator(img, line_width=self.outline_thickness)
        if result.masks is not None:
            masks = result.masks.xy
            for mask in masks:
                annotator.seg_bbox(
                    mask=mask, 
                    mask_color=self.outline_color, 
                    det_label=None
                )  
        return img     

    #############################################################################################################
    # CALL:

    def __call__(self, img):
        # Predict objects in image/frame
        results = self.predict(img)  
        # Read detections from image/frame
        detections = self.read_detections(results[0])

        # Draw segmentation masks
        if(self.show_show_mask):
            img = self.mask_annotator.annotate(scene=img, detections=detections)  

        # Draw outlines
        if(self.show_show_mask):
            img = self.annotate_outlines(img, results[0])

        # Draw bounding boxes with labels
        img = self.annotate_bboxes(img, detections)   

        return img 
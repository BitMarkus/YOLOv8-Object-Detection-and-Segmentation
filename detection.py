####################
# Object detection #
####################

import supervision as sv
# Own modules
from counter import Count
from model import ModelOD
from settings import setting

class Detect():

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self):
        # Load model and class names
        self.mdl = ModelOD()
        self.model = self.mdl.model
        self.class_names = self.mdl.class_names

        # Prediction parameters
        # List of classes to detect, empy list = all classes
        if(len(setting["od_class_selection"]) == 0):
            self.od_class_list = None
        else:
            self.od_class_list = setting["od_class_selection"]
        # Image size for inference
        self.inf_img_size = (setting["od_inf_img_size_width"], setting["od_inf_img_size_height"])
        # Sets the minimum confidence threshold for detections
        self.min_conf = setting["od_min_conf"]   
        # Limits the amount of detections on an image (default 300)
        self.max_detections = setting["od_max_detections"]  
        # Intersection Over Union (IoU) threshold
        self.iou = setting["od_iou"]  

        # Class counter object
        self.show_class_counter = setting["od_show_class_counter"]
        if(self.show_class_counter):
            self.class_counter = Count()

        # Annotations
        self.show_labels = setting["od_show_labels"]
        self.show_bbox = setting["od_show_bbox"]
        # Object for box annotations
        self.box_annotator = sv.BoundingBoxAnnotator(
            color=sv.ColorPalette.DEFAULT, 
            thickness=setting["od_bbox_line_thickness"], 
        )   
        # Object for label annotations   
        self.label_annotator = sv.LabelAnnotator(
            text_color=sv.Color.BLACK,
            text_thickness=setting["od_bbox_text_thickness"], 
            text_scale=setting["od_bbox_text_scale"]
        ) 

    #############################################################################################################
    # METHODS:

    # Predict on images/frames
    # https://docs.ultralytics.com/modes/predict/
    def predict(self, source):
        results = self.model.predict(
            source=source, 
            conf=self.min_conf,
            iou=self.iou,
            imgsz=self.inf_img_size,
            max_det=self.max_detections,
            classes=self.od_class_list
        )
        return results 
    
    # Read detections from image/frame
    def read_detections(self, result):
        return sv.Detections.from_ultralytics(result)
    
    def annotate_bboxes(self, img, detections):
        # Format custom labels
        self.labels = [f"{self.class_names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _
            in detections]
        
        # Annotate bounding boxes
        if(self.show_bbox):
            img = self.box_annotator.annotate(scene=img, detections=detections) # scene=frame.copy(),
        # Annotate labels
        if(self.show_labels):
            img = self.label_annotator.annotate(scene=img, detections=detections, labels=self.labels)

        return img 
    
    #############################################################################################################
    # CALL:

    def __call__(self, img):
        # Predict objects in image/frame
        results = self.predict(img)  

        # Read detections from image/frame
        detections = self.read_detections(results[0])

        # Draw bounding boxes with labels
        img = self.annotate_bboxes(img, detections)  
         
        # Show class counter
        if(self.show_class_counter):
            img = self.class_counter(results[0], img, self.class_names)

        return img 

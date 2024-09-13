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

    def __init__(self, mode):
        # Modes for image and video detection
        # Image size for inference: It is more comfortable to have different values for images and videos
        if(mode == 'img'):
            self.inf_img_size = setting["od_inf_size_img"]
            self.rectangular = setting["od_rectangular_img"]
            # Save bounding box results from predictions as txt file
            # Only for images, not for videos
            self.save_bb_results = setting["od_save_bb_results"]
        elif(mode == 'vid'):
            self.inf_img_size = setting["od_inf_size_vid"] 
            self.rectangular = setting["od_rectangular_video"]
            self.save_bb_results = False

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
        # Sets the minimum confidence threshold for detections
        self.min_conf = setting["od_min_conf"]   
        # Limits the amount of detections on an image (default 300)
        self.max_detections = setting["od_max_detections"]  
        # Intersection Over Union (IoU) threshold
        self.iou = setting["od_iou"]  
        # Path to training output
        self.pth_training_output = setting["pth_output"]

        # Class counter object
        self.show_class_counter = setting["od_show_class_counter"]
        if(self.show_class_counter):
            self.class_counter = Count()

        # Annotations
        self.show_labels = setting["od_show_labels"]
        self.show_bbox = setting["od_show_bbox"]
        # Object for box annotations
        self.box_annotator = sv.BoxAnnotator(
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
        if(self.save_bb_results):
            results = self.model.predict(
                #################################
                save_txt=self.save_bb_results,
                project=self.pth_training_output,
                name="tmp",
                #################################
                source=source, 
                conf=self.min_conf,
                iou=self.iou,
                imgsz=self.inf_img_size,
                rect=self.rectangular,
                max_det=self.max_detections,
                classes=self.od_class_list,
            )
        else:
            results = self.model.predict(
                source=source, 
                conf=self.min_conf,
                iou=self.iou,
                imgsz=self.inf_img_size,
                rect=self.rectangular,
                max_det=self.max_detections,
                classes=self.od_class_list,
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
            img = self.box_annotator.annotate(scene=img.copy(), detections=detections) # scene=frame.copy(),
        # Annotate labels
        if(self.show_labels):
            img = self.label_annotator.annotate(scene=img, detections=detections, labels=self.labels)

        return img 
    
    # Function formats the results for one image as a string
    # to save it in a text file as table
    def get_detection_result(self, result):
        # Detection results as dict
        det_result = {}
        # Read detections per image/frame and class
        # Check first if there are any detections in the image
        if(len(result.boxes)):
            # Iterate over class names
            for class_index, class_name in self.class_names.items():
                # Determine count per class
                object_count = result.boxes.cls.tolist().count(class_index)
                # Build dict with results (class: count)
                det_result[class_name] = object_count
        # If there are no detections
        else:
            # Iterate over class names
            for class_index, class_name in self.class_names.items():   
                # Detections for class is 0
                det_result[class_name] = 0  

        return det_result

    #############################################################################################################
    # CALL:

    def __call__(self, img):
        # Predict objects in image/frame
        results = self.predict(img)  

        result_dict = self.get_detection_result(results[0])

        # Read detections from image/frame
        detections = self.read_detections(results[0])

        # Draw bounding boxes with labels
        img = self.annotate_bboxes(img, detections)  
         
        # Show class counter
        if(self.show_class_counter):
            img = self.class_counter(results[0], img, self.class_names)

        return img, result_dict 

import cv2
from ultralytics import YOLO
import supervision as sv
# Own modules
from settings import setting

class ObjectDetection:

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self):
        # Path for prediction output
        self.pth_output = setting["pth_output"]
        # Select model for detection (pretrained vs. custom)
        self.use_pretrained_model = setting["od_use_pretrained_model"]
        # Load model
        # Pretrained model
        if(self.use_pretrained_model):
            self.model_name = f'{setting["od_model_prefix"]}{setting["od_pretrained_model_size"]}{setting["od_pretrained_model_extension"]}'
            # Model with path
            self.model_pth = f'{setting["pth_yolo_models"]}{self.model_name}'
        # Custom model
        else:
            self.model_name = setting["od_custom_model_name"]
            # Model with path
            self.model_pth = f'{setting["pth_custom_models"]}{self.model_name}'  
        self.model = self.load_model(self.model_pth)
        # Get a dict with all class names the model was trained on
        self.class_names = self.model.model.names 
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

        # Result window: show, position and title
        self.window_results_title = setting["window_results_title"]
        self.window_results_x_pos = setting["window_results_x_pos"]
        self.window_results_y_pos = setting["window_results_y_pos"] 

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

        # Class counter display
        self.show_class_counter = setting["od_show_class_counter"]
        self.counter_offset = setting["od_counter_offset"]
        self.counter_line_height = setting["od_counter_line_height"]
        self.counter_scale = setting["od_counter_scale"]
        self.counter_color = setting["od_counter_color"]
        self.counter_thickness = setting["od_counter_thickness"]

    #############################################################################################################
    # METHODS:

    def load_model(self, model_name):
        # load YOLOv8 model
        model = YOLO(model_name) 
        # fuse() optimizes the model by fusing Conv2d and BatchNorm2d layers, which can improve inference speed
        model.fuse
        return model
    
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
    
    def annotate_bboxes(self, result, img):
        # Setup detections for visualization
        detections = sv.Detections(
            xyxy = result.boxes.xyxy.cpu().numpy(),
            confidence = result.boxes.conf.cpu().numpy(),
            class_id = result.boxes.cls.cpu().numpy().astype(int)
        ) 
        # Format custom labels
        self.labels = [f"{self.class_names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _
            in detections]
        # Annotate frame
        if(self.show_bbox):
            img = self.box_annotator.annotate(scene=img, detections=detections) # scene=frame.copy(),
        if(self.show_labels):
            img = self.label_annotator.annotate(scene=img, detections=detections, labels=self.labels)
        if(self.show_class_counter):
            img = self.show_bbox_counter(result, img)
        return img 
    
    # Annotate images/frames with text
    def annotate_text(self, text, img, pos_x, pos_y):
        cv2.putText(
            img, 
            text, 
            (pos_x, pos_y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=self.counter_scale, 
            color=self.counter_color, 
            thickness=self.counter_thickness,
            lineType=cv2.LINE_AA,
        )
    
    # Print object counter onscreen while object detection
    # https://github.com/ultralytics/yolov5/issues/12947
    def show_bbox_counter(self, result, img):
        # Display header
        pos_x = self.counter_offset[0]
        pos_y = self.counter_offset[1] + self.counter_line_height
        self.annotate_text("COUNTS:", img, pos_x, pos_y)
        # Count lines for line height
        line_count = 2
        # Read detections per image/frame and class
        # Check first if there are any detections in the image
        if(len(result.boxes)):
            # Iterate over class names
            for class_index, class_name in self.class_names.items():
                # Determine count per class
                object_count = result.boxes.cls.tolist().count(class_index)
                # Only add class if count > 0
                if(object_count > 0):
                    # print(class_name, object_count)
                    label = f"{class_name}: {object_count}"
                    # Display lines
                    pos_x = self.counter_offset[0]
                    pos_y = self.counter_offset[1] + (line_count * self.counter_line_height)
                    self.annotate_text(label, img, pos_x, pos_y)
                    # increment line count
                    line_count += 1
        # In case there are no detections in the image
        else:
            pos_x = self.counter_offset[0]
            pos_y = self.counter_offset[1] + (line_count * self.counter_line_height)
            self.annotate_text("no detections", img, pos_x, pos_y)
        return img
    
    def print_classes(self):
        print("Model name:", self.model_pth)
        print("Number of classes:", len(self.class_names))
        for key, value in self.class_names.items():
            print(f"[{key}]: {value}")

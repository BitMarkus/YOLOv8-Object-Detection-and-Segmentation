######################
# Handling of models #
######################

from ultralytics import YOLO
# Own modules
from settings import setting

class ModelOD():

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self):
        # Select model for detection (pretrained vs. custom)
        self.use_pretrained_model = setting["od_use_pretrained_model"]
        # Load model
        # Pretrained model
        if(self.use_pretrained_model):
            # Normal vs. segmentation model
            if(setting["od_use_segmentation_model"]):
                self.model_name = f'{setting["od_model_prefix"]}{setting["od_pretrained_model_size"]}{setting["od_segmentation_model_postfix"]}{setting["od_pretrained_model_extension"]}'
            else:
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

    #############################################################################################################
    # METHODS:

    # Load model: Pretrained or custom trained
    def load_model(self, model_name):
        # load YOLOv8 model
        model = YOLO(model_name) 
        # fuse() optimizes the model by fusing Conv2d and BatchNorm2d layers, which can improve inference speed
        model.fuse
        return model
    
    def print_classes(self):
        print("Model name:", self.model_pth)
        print("Number of classes:", len(self.class_names))
        for key, value in self.class_names.items():
            print(f"[{key}]: {value}")
# Object Detection and Segmentation with Ultralytics YOLO
# Copyright (C) 2024 Markus Reichold <reichold.markus@gmx.de>

# This file is part of Object Detection and Segmentation with Ultralytics YOLO.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

######################
# Handling of models #
######################

import os
from ultralytics import YOLO
# Own modules
from settings import setting

class ModelOD():

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self, pth=None):

        # If no path to a specific model is given in the constructor,
        # the path/model is determined by the settings
        if(pth is None):
            # Select model for detection (pretrained vs. custom)
            self.use_pretrained_model = setting["od_use_pretrained_model"]
            # Model name strings
            self.model_prefix = "yolov8"
            self.segmentation_model_postfix = "-seg"
            self.pretrained_model_extension = ".pt"
            # Load model
            # Pretrained model
            if(self.use_pretrained_model):
                # Normal vs. segmentation model
                if(setting["od_use_segmentation_model"]):
                    self.model_name = f'{self.model_prefix}{setting["od_pretrained_model_size"]}{self.segmentation_model_postfix}{self.pretrained_model_extension}'
                else:
                    self.model_name = f'{self.model_prefix}{setting["od_pretrained_model_size"]}{self.pretrained_model_extension}'
                # Model with path
                self.model_pth = f'{setting["pth_yolo_models"]}{self.model_name}'
            # Custom model
            else:
                self.model_name = setting["od_custom_model_name"]
                # Model with path
                self.model_pth = f'{setting["pth_custom_models"]}{self.model_name}'  

        # If a path to a model is specified in the constructor,
        # the settings will be ignored and the model in the specified path will be loaded
        else:
            # Get model name from path
            # https://stackoverflow.com/questions/3925096/how-to-get-only-the-last-part-of-a-path-in-python
            self.model_pth = pth
            self.model_name = os.path.basename(os.path.normpath(self.model_pth))

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
    
    # Method to print the classes of a model
    def print_classes(self):
        print("Model name:", self.model_pth)
        print("Number of classes:", len(self.class_names))
        for key, value in self.class_names.items():
            print(f"[{key}]: {value}")
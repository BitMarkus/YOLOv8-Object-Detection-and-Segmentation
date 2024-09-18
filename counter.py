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

##########################
# Onscreen Class Counter #
##########################

# Own modules
import functions as fcn
from settings import setting

class Count():

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self):
        self.counter_offset = setting['od_counter_offset']
        self.counter_line_height = setting['od_counter_line_height'] 
        self.counter_scale = setting["od_counter_scale"]  
        self.counter_color = setting["od_counter_color"] 
        self.counter_thickness = setting["od_counter_thickness"] 

    #############################################################################################################
    # CALL:

    # Print object counter onscreen while object detection
    # https://github.com/ultralytics/yolov5/issues/12947
    def __call__(self, result, img, class_names):
        
        # Display header
        pos = (self.counter_offset[0], self.counter_offset[1] + self.counter_line_height)
        fcn.annotate_text(
            img, 
            "COUNTS:", 
            pos, 
            self.counter_scale,
            self.counter_color,
            self.counter_thickness,
        )

        # Count lines for line height
        line_count = 2
        # Read detections per image/frame and class
        # Check first if there are any detections in the image
        if(len(result.boxes)):
            # Iterate over class names
            for class_index, class_name in class_names.items():
                # Determine count per class
                object_count = result.boxes.cls.tolist().count(class_index)
                # Only add class if count > 0
                if(object_count > 0):
                    # print(class_name, object_count)
                    label = f"{class_name}: {object_count}"
                    # Display lines
                    pos = (self.counter_offset[0], self.counter_offset[1] + (line_count * self.counter_line_height))
                    fcn.annotate_text(
                        img, 
                        label, 
                        pos,
                        self.counter_scale,
                        self.counter_color,
                        self.counter_thickness,                        
                    )
                    # increment line count
                    line_count += 1

        # In case there are no detections in the image
        else:
            pos = (self.counter_offset[0], self.counter_offset[1] + (line_count * self.counter_line_height))
            fcn.annotate_text(
                img, 
                "no detections",
                pos,
                self.counter_scale,
                self.counter_color,
                self.counter_thickness,  
            )

        return img    
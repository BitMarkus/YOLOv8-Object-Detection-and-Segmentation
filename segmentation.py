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

#######################
# Object segmentation #
#######################

from ultralytics.utils.plotting import Annotator, colors
import supervision as sv
# Own modules
from counter import Count
from detection import Detect
from settings import setting

class Segment(Detect):

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, mode, model_object):
        super().__init__(mode, model_object)

        # Mode for images or videos
        # Important for setting inferance size for basis class
        self.mode = mode

        # Outline thickness and color of segmentation masks
        self.show_show_mask = setting["os_show_mask"]  
        self.show_outline = setting["os_show_outline"]  
        self.outline_thickness = setting["os_outline_thickness"]  
        self.outline_color = setting["os_outline_color"] 

        # Class counter object
        self.show_class_counter = setting["od_show_class_counter"]
        if(self.show_class_counter):
            self.class_counter = Count() 

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
                )  
        return img     

    #############################################################################################################
    # CALL:

    def __call__(self, img, save_bb_txt=False):
        # Predict objects in image/frame
        results = self.predict(img, save_bb_txt)  
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

        # Show class counter
        if(self.show_class_counter):
            img = self.class_counter(results[0], img, self.class_names)

        return img 
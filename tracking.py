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

################################
# Object tracking (for videos) #
################################

from ultralytics import solutions
# Own modules
from model import ModelOD
from settings import setting

class Track():

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self):
        # Load model and class names
        self.mdl = ModelOD()
        self.model = self.mdl.model
        self.class_names = self.mdl.class_names

        # List of classes to count/track, empy list = all classes
        if(len(setting["oc_class_selection"]) == 0):
            self.class_list = None
        else:
            self.class_list = setting["oc_class_selection"]   
        # Confidence for counting/tracking
        self.min_conf = setting["oc_min_conf"]  
        # Calculate coordinates of counter region
        # x,y-coordinares: Upper left point, upper right point, bottom left point, bottom right point
        roi_count = [
            (setting["oc_roi_start"][0], setting["oc_roi_start"][1]), 
            (setting["oc_roi_start"][0] + setting["oc_roi_size"][0], setting["oc_roi_start"][1]), 
            (setting["oc_roi_start"][0], setting["oc_roi_start"][1] + setting["oc_roi_size"][1]), 
            (setting["oc_roi_start"][0] + setting["oc_roi_size"][0], setting["oc_roi_start"][1] + setting["oc_roi_size"][1])]
        # Init Object Counter
        # https://docs.ultralytics.com/guides/object-counting/#argument-objectcounter 
        # https://github.com/orgs/ultralytics/discussions/6738:
        # InCount: Object that pass the region from right to left or bottom to top
        # OutCount: Object that pass the region from left to right or top to bottom 
        self.counter = solutions.ObjectCounter(
            view_img=False,
            reg_pts=roi_count,
            names=self.class_names,
            view_in_counts=setting["oc_view_in_counts"],
            view_out_counts=setting["oc_view_out_counts"],
            draw_tracks=setting["oc_draw_tracks"],
            line_thickness=setting["oc_bboxes_line_thickness"],  
            # Not supported anymore:
            # track_thickness=setting["oc_track_thickness"],           
            # count_reg_color=setting["oc_box_color"],
            # region_thickness=setting["oc_box_line_thickness"],
            # count_txt_color=setting["oc_text_color"],
            # count_bg_color=setting["oc_bg_color"],
        )

    #############################################################################################################
    # CALL:

    def __call__(self, frame):
        tracks = self.model.track(
            source=frame, 
            persist=True, 
            show=False,
            conf=self.min_conf,
            classes=self.class_list,
        )
        frame = self.counter.start_counting(frame, tracks)
        return frame
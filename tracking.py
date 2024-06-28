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
            count_txt_color=setting["oc_text_color"],
            count_bg_color=setting["oc_bg_color"],
            classes_names=self.class_names,
            view_in_counts=setting["oc_view_in_counts"],
            view_out_counts=setting["oc_view_out_counts"],
            draw_tracks=setting["oc_draw_tracks"],
            track_thickness=setting["oc_track_thickness"],           
            count_reg_color=setting["oc_box_color"],
            region_thickness=setting["oc_box_line_thickness"],
            line_thickness=setting["oc_bboxes_line_thickness"],  
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
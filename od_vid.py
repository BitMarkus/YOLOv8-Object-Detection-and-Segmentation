###############################
# Object detection for videos #
###############################

import numpy as np
import cv2
from time import time
# Own modules
from od import ObjectDetection
from vid_oc import ObjectCountingTracking
from settings import setting

class ObjectDetectionVid(ObjectDetection):

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self, mode):
        super().__init__()

        self.source = setting["od_video_source"]      
        self.show_class_counter = setting["od_show_class_counter"]  
        self.show_output_video = setting["od_show_output_video"]  # -> NO ESC IF FALSE
        # width and hight of webcam image for display
        self.frame_width = setting["od_frame_width"]  
        self.frame_height = setting["od_frame_height"] 
        # FPS display
        self.show_fps = setting["show_fps"]
        self.fps_position = setting["fps_position"]
        self.fps_font_scale = setting["fps_font_scale"]
        self.fps_color = setting["fps_color"]
        self.fps_font_thickness = setting["fps_font_thickness"]
        # Object counter/tracker
        if(mode == 'detect'):
            self.activate_object_detection = True
            self.activate_object_counting = False
        elif(mode == 'count'):
            self.activate_object_detection = False
            self.activate_object_counting = True
            self.object_counter_tracker = ObjectCountingTracking()

    #############################################################################################################
    # METHODS:

    # Shows the result window i.g. webcam stream
    def show_predicted_frame(self, frame, first_frame):
        cv2.namedWindow(self.window_results_title)
        if(first_frame):
            cv2.moveWindow(self.window_results_title, self.window_results_x_pos, self.window_results_y_pos)
        cv2.imshow(self.window_results_title, frame)

    def fps_counter(self, frame, start, end):
        # Calculate FPS
        fps = 1/np.round(end - start, 2)
        # Put text into frame
        ObjectDetection.annotate_text(self, f'FPS: {int(fps)}', frame, self.fps_position[0], self.fps_position[1])

    #############################################################################################################
    # CALL:

    def __call__(self):
        # Capture video
        cap = cv2.VideoCapture(self.source)
        assert cap.isOpened()

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        # Setting dependent on webcam model (only for linux):
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'X264'))

        # For the first frame, the result window is positioned
        # Or else it appears randomly all over the screen
        first_frame = True

        while True:
            # Start fps counter
            if(self.show_fps):
                start_time = time()

            # Read frame
            ret, frame = cap.read()
            assert ret

            # Object detection
            if(self.activate_object_detection):
                # Predict objects in frame
                results = ObjectDetection.predict(self, frame)
                # Plot bounding boxes in frame
                frame = ObjectDetection.annotate_bboxes(self, results[0], frame)

            # Object counting/tracking
            if(self.activate_object_counting):
                self.object_counter_tracker(frame)

            # End fps counter and show fps in left upper corner
            if(self.show_fps):
                end_time = time()
                self.fps_counter(frame, start_time, end_time)

            # Show frame with cv2
            if(self.show_output_video):
                self.show_predicted_frame(frame, first_frame)

            # Set first frame to false
            first_frame = False
            
            # Press ESC to end program
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
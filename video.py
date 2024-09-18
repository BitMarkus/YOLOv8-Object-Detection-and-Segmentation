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

###################################
# Handling of videos, e.g. webcam #
###################################

import cv2
import numpy as np
from time import time
# Own modules
from detection import Detect
from model import ModelOD
from segmentation import Segment
from tracking import Track
from settings import setting
import functions as fcn

class VideoOD():

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self, mode):
        # Modes for object detection on videos:
        # 1) 'detect': Object detection with bounding boxes
        # 2) 'segment': Instance segmentation with masks (can also include bounding boxes)
        # 3) 'track': Object tracking and counting 
        if(mode == 'detect'):
            self.activate_object_detection = True
            self.activate_object_segmentation = False
            self.activate_object_tracking = False        
        elif(mode == 'segment'):
            self.activate_object_detection = False
            self.activate_object_segmentation = True 
            self.activate_object_tracking = False 
        elif(mode == 'track'):
            self.activate_object_detection = False
            self.activate_object_segmentation = False
            self.activate_object_tracking = True

        # Source: Webcam (0) or path to video file
        self.source = setting["od_video_source"]  
        # Show output video with cv2
        self.show_output_video = setting["od_show_output_video"]
        # Save output video in output folder
        self.save_output_video = setting["od_save_output_video"]
        # Name of the saved video file
        self.output_video_name = setting["od_output_video_name"] 
        # Width, hight and fps of video for display
        self.frame_width = setting["od_frame_width"]  
        self.frame_height = setting["od_frame_height"] 
        self.frame_fps = setting["od_frame_fps"] 
        # Saved video frames (by pressing s key)
        # Path for saving
        self.pth_video_frames = setting["pth_video_frames"]
        # Output path for saved videos
        self.pth_video_output = setting["pth_output"]
        # Name and extension of saved video frames
        self.saved_video_frame_prefix = setting["od_saved_video_frame_prefix"]
        self.saved_video_frame_extension = setting["od_saved_video_frame_extension"]
        # Result window: show, position and title
        self.window_results_title = setting["window_results_title"]
        self.window_results_x_pos = setting["window_results_x_pos"]
        self.window_results_y_pos = setting["window_results_y_pos"] 
        # FPS display
        self.show_fps_display = setting["show_fps"]
        self.fps_position = setting["fps_position"]
        self.fps_font_scale = setting["fps_font_scale"]
        self.fps_color = setting["fps_color"]
        self.fps_font_thickness = setting["fps_font_thickness"]

        # Load model and class names for prediction
        model_object = ModelOD()

        # Modes
        # Object detection
        if(self.activate_object_detection):
            self.detection = Detect('vid', model_object)
        # Object segmentation
        if(self.activate_object_segmentation):
            self.segmentation = Segment('vid', model_object) 
        # Object tracking
        if(self.activate_object_tracking):
            self.tracker = Track()       

    #############################################################################################################
    # METHODS:

    # Shows the result window i.g. webcam stream
    def show_frame(self, frame, first_frame):
        cv2.namedWindow(self.window_results_title)
        if(first_frame):
            cv2.moveWindow(self.window_results_title, self.window_results_x_pos, self.window_results_y_pos)
        cv2.imshow(self.window_results_title, frame)

    def fps_display(self, frame, start, end):
        # Calculate FPS
        fps = 1/np.round(end - start, 2)
        # Put text into frame
        fcn.annotate_text(
            frame, 
            f'FPS: {int(fps)}', 
            self.fps_position, 
            self.fps_font_scale, 
            self.fps_color, 
            self.fps_font_thickness,
        )

    #############################################################################################################
    # CALL:

    def __call__(self):

        # Capture video
        cap = cv2.VideoCapture(self.source)
        assert cap.isOpened()

        # Set video format (width, height, fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.frame_fps)
        # Setting dependent on webcam model (only for linux):
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'X264'))

        # Save output video
        if(self.save_output_video):
            w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(f'{self.pth_video_output}{self.output_video_name}.avi', cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        # For the first frame, the result window is positioned
        # Or else it appears randomly all over the screen
        first_frame = True
        # Image conter (for video screenshot savings)
        img_counter = 0

        while True:
            # Start fps counter
            if(self.show_fps_display):
                start_time = time()

            # Read frame
            ret, frame = cap.read()
            assert ret

            # OBJECT DETECTION #
            if(self.activate_object_detection):
                frame, _ = self.detection(frame)

            # OBJECT SEGMENTATION #
            if(self.activate_object_segmentation):
                frame = self.segmentation(frame)

            # OBJECT TRACKING #
            if(self.activate_object_tracking):
                frame = self.tracker(frame)

            # End fps counter and show fps in left upper corner
            if(self.show_fps_display):
                end_time = time()
                self.fps_display(frame, start_time, end_time)

            # Save output video
            if(self.save_output_video):
                out.write(frame)

            # Show frame with cv2
            if(self.show_output_video):
                self.show_frame(frame, first_frame)

            # Set first frame to false
            first_frame = False
            
            # Keypress actions while video is showing:
            k = cv2.waitKey(1)
            # Press ESC to end live window
            if(k%256 == 27):
                print("Esc was pressed. Closing live window!")
                break
            # Press s key to save frame
            elif(k%256 == ord('s')):
                img_name = f'{self.pth_video_frames}{self.saved_video_frame_prefix}{img_counter}{self.saved_video_frame_extension}'
                cv2.imwrite(img_name, frame)
                print(f'Video frame saved as {img_name}!')
                img_counter += 1

        if(self.save_output_video):
            out.release()
        cap.release()
        cv2.destroyAllWindows()
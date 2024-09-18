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

#################################
# Handling of screen recordings #
#################################

import mss
from time import time
import cv2
import numpy as np
from PIL import Image
# Own modules
from video import VideoOD
from settings import setting

class ScreenOD(VideoOD):

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, mode):
        super().__init__(mode)

        # Monitor selection for multi monitor setup
        self.monitor_number = setting["od_monitor_number"]  
        # Offset of capture region from the left upper corner of the monitor
        self.capture_roi_offset_xy = setting["od_capture_roi_offset_xy"]
        # Width and height of the capture region
        self.capture_roi_size_wh = setting["od_capture_roi_size_wh"]
        # FPS for capture video recording
        self.capture_roi_record_fps = setting["od_capture_roi_record_fps"]

    #############################################################################################################
    # CALL:

    def __call__(self):

        with mss.mss() as sct:
            # Read monitor image
            mon = sct.monitors[self.monitor_number]
            monitor = {
                "top": mon["top"] + self.capture_roi_offset_xy[1],
                "left": mon["left"] + self.capture_roi_offset_xy[0],
                "width": self.capture_roi_size_wh[0],
                "height": self.capture_roi_size_wh[1],
                "mon": self.monitor_number,
            }

            # Save output video
            if(self.save_output_video):
                out = cv2.VideoWriter(
                    f'{self.pth_video_output}{self.output_video_name}.avi', 
                    cv2.VideoWriter_fourcc(*"MJPG"), 
                    self.capture_roi_record_fps, 
                    (self.capture_roi_size_wh[0], self.capture_roi_size_wh[1]),
                )

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
                frame = np.array(sct.grab(monitor))
                # print(frame.shape)
                if len(frame.shape) > 2 and frame.shape[2] == 4:
                    #convert the image from RGBA2RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # COLOR_BGRA2RGB
     
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
            cv2.destroyAllWindows()
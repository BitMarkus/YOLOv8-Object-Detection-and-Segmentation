###############################
# Object detection for videos #
###############################

import numpy as np
import cv2
from time import time
import supervision as sv
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
        self.show_output_video = setting["od_show_output_video"]
        self.save_output_video = setting["od_save_output_video"]
        self.output_video_name = setting["od_output_video_name"]
        # width and hight of webcam image for display
        self.frame_width = setting["od_frame_width"]  
        self.frame_height = setting["od_frame_height"] 
        self.frame_fps = setting["od_frame_fps"] 
        # FPS display
        self.show_fps = setting["show_fps"]
        self.fps_position = setting["fps_position"]
        self.fps_font_scale = setting["fps_font_scale"]
        self.fps_color = setting["fps_color"]
        self.fps_font_thickness = setting["fps_font_thickness"]
        # Paths
        self.pth_video_frames = setting["pth_video_frames"]
        # Object detection, counter/tracker or segmentation
        if(mode == 'detect'):
            self.activate_object_detection = True
            self.activate_object_counting = False
            self.activate_object_segmentation = False
        elif(mode == 'count'):
            self.activate_object_detection = False
            self.activate_object_counting = True
            self.activate_object_segmentation = False
            self.object_counter_tracker = ObjectCountingTracking()
        elif(mode == 'segm'):
            self.activate_object_detection = False
            self.activate_object_counting = False
            self.activate_object_segmentation = True

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
            out = cv2.VideoWriter(f'{self.pth_output}{self.output_video_name}.avi', cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

        # For the first frame, the result window is positioned
        # Or else it appears randomly all over the screen
        first_frame = True
        # Image conter (for webcam screenshot savings)
        img_counter = 0

        while True:
            # Start fps counter
            if(self.show_fps):
                start_time = time()

            # Read frame
            ret, frame = cap.read()
            assert ret

            # Object detection
            if(self.activate_object_detection or self.activate_object_segmentation):
                # Predict objects in frame
                results = ObjectDetection.predict(self, frame)
                # Plot bounding boxes in frame
                frame = ObjectDetection.annotate_bboxes(self, self.activate_object_detection, self.activate_object_segmentation, results[0], frame)

            # Object counting/tracking
            if(self.activate_object_counting):
                self.object_counter_tracker(frame)

            # End fps counter and show fps in left upper corner
            if(self.show_fps):
                end_time = time()
                self.fps_counter(frame, start_time, end_time)

            # Save output video
            if(self.save_output_video):
                out.write(frame)

            # Show frame with cv2
            if(self.show_output_video):
                self.show_predicted_frame(frame, first_frame)

            # Set first frame to false
            first_frame = False
            
            # Keypress actions:
            k = cv2.waitKey(1)
            # Press ESC to end program
            if(k%256 == 27):
                print("Esc was pressed. Closing application!")  # if cv2.waitKey(5) & 0xFF == 27:
                break
            # Press s key to save frame
            elif(k%256 == ord('s')):
                img_name = f'{self.pth_video_frames}camshot_frame_{img_counter}.png'
                cv2.imwrite(img_name, frame)
                print(f'Camshot {img_name} successfully saved!')
                img_counter += 1


        if(self.save_output_video):
            out.release()
        cap.release()
        cv2.destroyAllWindows()
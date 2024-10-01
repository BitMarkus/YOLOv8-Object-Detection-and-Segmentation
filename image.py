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
# Handling of images #
######################

import cv2
from PIL import Image
import os
import shutil
# Own modules
from detection import Detect
from model import ModelOD
from segmentation import Segment
from settings import setting

class ImageOD():

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self, mode):
        # Modes for object detection on images:
        # 1) 'detect': Object detection with bounding boxes
        # 2) 'segment': Instance segmentation with masks (can also include bounding boxes)
        if(mode == 'detect'):
            self.activate_object_detection = True
            self.activate_object_segmentation = False
            self.activate_object_tracking = False 
            self.save_bb_txt = True       
        elif(mode == 'segment'):
            self.activate_object_detection = False
            self.activate_object_segmentation = True 
            self.activate_object_tracking = False 
            self.save_bb_txt = False           

        # Path for images to predict
        self.pth_predictions = setting["pth_predictions"]
        # Output path for saved images
        self.pth_image_output = setting["pth_output"]
        # Show predicted images
        self.show_pred_images = setting["od_show_predicted_images"]  
        # Save predicted images
        self.save_pred_images = setting["od_save_predicted_images"] 
        # Result window: show, position and title
        self.window_results_title = setting["window_results_title"]
        self.window_results_x_pos = setting["window_results_x_pos"]
        self.window_results_y_pos = setting["window_results_y_pos"] 
        # Result export
        self.export_results = setting["od_export_results"] 
        # Export results: file name
        self.export_file_name = setting["od_export_file_name"] 
        # Parameters for predictions (for result export)
        self.iou = setting["od_iou"]
        self.min_conf = setting["od_min_conf"]
        # Model name
        self.model_name = None
        self.model_pth = None
        # Save bounding box results from predictions as txt file
        self.save_bb_results = setting["od_save_bb_results"] 

    #############################################################################################################
    # METHODS:

    # Returns a list with all images (files, not folders) in the predictions folder
    def get_img_list(self, pth):
        images = [img for img in os.listdir(pth) if os.path.isfile(os.path.join(pth, img))]
        return images
    
    # Load image
    def load_image(self, name, pth):
        img_pth = f'{pth}{name}'
        img = cv2.imread(img_pth)
        return img

    # Saves predicted images
    def save_image(self, img, name, output_pth):
        img = Image.fromarray(img[:,:,::-1]) # Change color to rgb
        img_pth = f'{output_pth}{name}'
        img.save(img_pth)
        print(f'Image {name} was saved to folder {output_pth}.')

    # Displays the (predicted) image
    def show_image(self, img):
        cv2.namedWindow(self.window_results_title)
        cv2.moveWindow(self.window_results_title, self.window_results_x_pos, self.window_results_y_pos)
        cv2.imshow(self.window_results_title, img)
        # Close image window by pressing any key
        cv2.waitKey(0) 
        # closing all open windows
        cv2.destroyAllWindows()  

    # Create a text file for saving detection results
    def save_result_file(self, results, output_pth):

        # This mode opens the file for writing only. The data in existing files are modified and overwritten
        # If the file does not already exist in the folder, a new one gets created
        file_pth = f"{output_pth}{self.export_file_name}"
        with open(file_pth, "w") as f:

            # Get forst index in result dict
            first_idx = next(iter(results))
            # Get number of images and classes
            num_classes = len(results[first_idx])
            # num_img = len(results)

            # Write hyperparameters for prediction
            f.write("----------- PREDICTION PARAMETERS -----------\n")
            f.write("\n")
            f.write(f"Checkpoint: {self.model_pth}\n") 
            f.write(f"Min conf: {self.min_conf}\n") 
            f.write(f"IoU: {self.iou}\n")
            f.write("\n") 

            # Write detections table
            f.write("----------- DETECTIONS -----------\n")
            f.write("\n")
            # Write header
            f.write("Image name, ") 
            # Iterate over classes = first entry in dict results
            # Get first index in dict
            counter = 0 
            for class_name in results[first_idx]:
                f.write(f"Num {class_name}") 
                counter += 1
                if(counter < num_classes):
                    f.write(", ") 
                else:
                    f.write(":\n")  

            # Write table
            for img in results:
                # Image name
                f.write(f"{img}, ")
                # Number of detections per class
                counter = 0
                for class_name in results[img]:
                    f.write(f"{results[img][class_name]}")
                    counter += 1
                    if(counter < num_classes):
                        f.write(", ") 
                    else:
                        f.write("\n")                     

    # Copy label output in specified folder
    # and match txt file name with image name
    def generate_labels(self, image_name, output_pth):
        # Move labels file from temp folder to output folder
        src = f"{self.pth_image_output}tmp/labels/image0.txt"
        # Check, if the file exists
        if(os.path.exists(src)):
            shutil.move(src, output_pth)
            # Remove extension from image name
            image_name = os.path.splitext(image_name)[0]
            # Rename txt file to match image name
            os.rename(f"{output_pth}image0.txt", f"{output_pth}/{image_name}.txt")
        # Delete tmp folder
        # https://sentry.io/answers/delete-a-file-or-folder-in-python/
        src = f"{self.pth_image_output}tmp/"
        shutil.rmtree(src)

    #############################################################################################################
    # CALL:

    def __call__(self):  

        # Load model and class names for prediction
        model_object = ModelOD()
        self.model_pth = model_object.model_pth

        # Modes
        # Object detection
        if(self.activate_object_detection):
            self.detection = Detect('img', model_object)
        # Object segmentation
        if(self.activate_object_segmentation):
            self.segmentation = Segment('img', model_object)     

        # Load images in prediction folder
        images = self.get_img_list(self.pth_predictions)
        # print(images)

        # If there are any images in the folder
        if(len(images) > 0):
            
            # Create a dict to store class counts
            if(self.export_results):
                results = {}

            # Iterate over images
            for image_name in images:
                # Load image
                img = self.load_image(image_name, self.pth_predictions)
                # print(img)

                # OBJECT DETECTION #
                if(self.activate_object_detection):
                    img, result = self.detection(img, self.save_bb_txt)
                    # Add results to class count dict
                    results[image_name] = result

                    # Copy label output in output folder
                    if(self.save_bb_results and self.save_bb_txt):
                        self.generate_labels(image_name, self.pth_image_output)

                # OBJECT SEGMENTATION #
                if(self.activate_object_segmentation):
                    img = self.segmentation(img, self.save_bb_txt)

                # Show image
                if(self.show_pred_images):
                    self.show_image(img)

                # Save image
                if(self.save_pred_images):
                    self.save_image(img, image_name, self.pth_image_output)

            # Save results in a text file
            if(self.export_results and self.save_bb_txt):
                self.save_result_file(results, self.pth_image_output)

        # If there are no images in the prediction folder      
        else:
            print(f'No images were found in folder {self.pth_predictions}!')
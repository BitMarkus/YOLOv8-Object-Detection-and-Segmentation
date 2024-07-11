######################
# Handling of images #
######################

import cv2
from PIL import Image
import os
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
        elif(mode == 'segment'):
            self.activate_object_detection = False
            self.activate_object_segmentation = True 
            self.activate_object_tracking = False         

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
        # Export results: Path
        self.export_pth = f"{self.pth_image_output}{self.export_file_name}"
        # Parameters for predictions (for result export)
        self.iou = setting["od_iou"]
        self.min_conf = setting["od_min_conf"]
        # Load model and class names
        self.mdl = ModelOD()

        # Modes
        # Object detection
        if(self.activate_object_detection):
            self.detection = Detect('img')
        # Object segmentation
        if(self.activate_object_segmentation):
            self.segmentation = Segment('img')        

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
    def save_image(self, img, name, pth_output):
        img = Image.fromarray(img[:,:,::-1]) # Change color to rgb
        output_pth = f'{pth_output}{name}'
        img.save(output_pth)
        print(f'Image {name} was saved to folder {pth_output}.')

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
    def save_result_file(self, results):
        # This mode opens the file for writing only. The data in existing files are modified and overwritten
        # If the file does not already exist in the folder, a new one gets created
        f = open(self.export_pth, "w")

        # Get forst index in result dict
        first_idx = next(iter(results))
        # Get number of images and classes
        num_classes = len(results[first_idx])
        # num_img = len(results)

        # Write hyperparameters for prediction
        f.write("----------- PREDICTION PARAMETERS -----------\n")
        f.write("\n")
        f.write(f"Model name: {self.mdl.model_name}\n") 
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
                       
        # Close file
        f.close()

    #############################################################################################################
    # CALL:

    def __call__(self):
        # Load images in prediction folder
        images = self.get_img_list(self.pth_predictions)

        # If there are any images in the folder
        if(len(images) > 0):
            
            # Create a dict to store class counts
            if(self.export_results):
                results = {}

            # Iterate over images
            for image_name in images:
                # Load image
                img = self.load_image(image_name, self.pth_predictions)

                # OBJECT DETECTION #
                if(self.activate_object_detection):
                    img, result = self.detection(img)
                    results[image_name] = result

                # OBJECT SEGMENTATION #
                if(self.activate_object_segmentation):
                    img = self.segmentation(img)

                # Show image
                if(self.show_pred_images):
                    self.show_image(img)

                # Save image
                if(self.save_pred_images):
                    self.save_image(img, image_name, self.pth_image_output)

            # Save results in a text file
            if(self.export_results):
                print(results)
                self.save_result_file(results)

        # If there are no images in the prediction folder      
        else:
            print(f'No images were found in folder {self.pth_predictions}!')
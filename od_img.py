###############################
# Object detection for images #
###############################

import cv2
from PIL import Image
import os
# Own modules
from od import ObjectDetection
from settings import setting

class ObjectDetectionImg(ObjectDetection):

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self):
        super().__init__()

        # Path for images to predict
        self.pth_predictions = setting["pth_predictions"]
        # Show predicted images
        self.show_pred_images = setting["od_show_predicted_images"]  
        # Save predicted images
        self.save_pred_images = setting["od_save_predicted_images"] 

    #############################################################################################################
    # METHODS:

    # Shows prediction result on an image
    def show_predicted_images(self, results):
        # Iterate over results from image predictions
        for result in results:
            img = result.plot(labels=False, probs=False, conf=False, boxes=False, show=False)
            img = ObjectDetection.annotate_bboxes(self, result, img)
            cv2.namedWindow(self.window_results_title)
            cv2.moveWindow(self.window_results_title, self.window_results_x_pos, self.window_results_y_pos)
            cv2.imshow(self.window_results_title, img)
            # Waits for user to press ANY key 
            # (this is necessary to avoid Python kernel form crashing) 
            cv2.waitKey(0) 
            # closing all open windows 
            cv2.destroyAllWindows() 

    # Saves predicted images
    def save_predicted_images(self, results):
        print("\n>> Saving predicted images...")
        # Iterate over results from image predictions
        for result in results:
            # Save image with bounding boxes
            # https://docs.ultralytics.com/modes/predict/#plotting-results
            img = result.plot(labels=False, probs=False, conf=False, boxes=False, show=False)
            img = ObjectDetection.annotate_bboxes(self, result, img)
            img = Image.fromarray(img[:,:,::-1]) # Change color to rgb
            img_name = os.path.basename(result.path).split('/')[-1]
            result_pth = f"{self.pth_output}{img_name}"
            img.save(result_pth)
        print(f"Images are saved to folder {self.pth_output}.")

    #############################################################################################################
    # CALL:

    def __call__(self):
        # Predict on image(s)
        results = ObjectDetection.predict(self, self.pth_predictions)
        # Show images
        if(self.show_pred_images):
            self.show_predicted_images(results)
        # Save images
        if(self.save_pred_images):
            self.save_predicted_images(results)
######################
# Handling of images #
######################

import cv2
from PIL import Image
import os
# Own modules
from settings import setting

class ImageOD():

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self):
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

    #############################################################################################################
    # CALL:

    def __call__(self):
        # Load images in prediction folder
        images = self.get_img_list(self.pth_predictions)
        # If there are any images in the folder
        if(len(images) > 0):
            # Show images
            if(self.show_pred_images):
                # Iterate over images
                for image_name in images:
                    # Load image
                    img = self.load_image(image_name, self.pth_predictions)

                    # Show image
                    cv2.namedWindow(self.window_results_title)
                    cv2.moveWindow(self.window_results_title, self.window_results_x_pos, self.window_results_y_pos)
                    cv2.imshow(self.window_results_title, img)

                    # Close image window by pressing any key
                    cv2.waitKey(0) 
                    # closing all open windows
                    cv2.destroyAllWindows() 

            # Save images
            if(self.save_pred_images):
                # Iterate over images
                for image_name in images:
                    self.save_image(img, image_name, self.pth_image_output)

        # If there are no images in the prediction folder      
        else:
            print(f'No images were found in folder {self.pth_predictions}!')
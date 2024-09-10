########################################################
# Batch testing of object detection models/checkpoints #
########################################################

import os
from pathlib import Path
# Own modules
from model import ModelOD
from image import ImageOD
from settings import setting

class Batch_Pred(ImageOD):

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, mode='detect'):
        super().__init__(mode)

        # Path to train folders
        # This is NOT the path to checkpoints/models! It is the absolute path to any folder
        # This is the path to a folder which contains train folders from trainings
        # The actual checkpoints are inside a train folder in the folder weights
        self.pth_batch_test = setting["pth_batch_test"]

    #############################################################################################################
    # METHODS:

    # Returns a list with all images (files, not folders) in the predictions folder
    def get_folder_list(self, pth):
        dir_list = [f for f in os.listdir(pth) if os.path.isdir(os.path.join(pth, f))]
        return dir_list
    
    # Returns a list with all checkpoints in the specified folder
    def get_checkpoint_list(self, pth):
        checkpoint_list = [c for c in os.listdir(pth) if 
                    os.path.isfile(os.path.join(pth, c)) and
                    c.endswith('.pt')]
        return checkpoint_list

    #############################################################################################################
    # CALL:

    def __call__(self):

        # Load images in prediction folder
        images = self.get_img_list(self.pth_predictions)
        # If there are any images in the prediction folder
        if(len(images) > 0):
            print(f"There are {len(images)} images for prediction in the prediction folder.")

            # Check if there are train folders in the batch folder
            dir_list = self.get_folder_list(self.pth_batch_test)
            if(len(dir_list) > 0):
                print(f"There are {len(dir_list)} folders in {self.pth_predictions}.")

                # Iterate over folder list
                for dir_name in dir_list:
                    # Check, if the training folder contains a folder called "weights"
                    checkpoint_pth = f"{self.pth_batch_test}/{dir_name}/weights/"
                    if(os.path.exists(checkpoint_pth)):

                        # Get a list of all checkpoints in the weights folder
                        checkpoint_list = self.get_checkpoint_list(checkpoint_pth)
                        if(len(checkpoint_list) > 0):
                            
                            # Create a folder for the training folder
                            result_dir = f"{self.pth_image_output}{dir_name}/"
                            Path(result_dir).mkdir(parents=True, exist_ok=True)

                            # Iterate over checkpoint list
                            for checkpoint in checkpoint_list:

                                # Load checkpoint
                                self.mdl = ModelOD(f"{checkpoint_pth}{checkpoint}")

                                # Create a directory with the checkpoint name
                                checkpoint_dir = f"{result_dir}{os.path.splitext(checkpoint)[0]}/"
                                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

                                # Create a dict to store class counts
                                if(self.export_results):
                                    results = {}      

                                # Iterate over images
                                for image_name in images:
                                    
                                    # Load image
                                    img = self.load_image(image_name, self.pth_predictions)

                                    # Perform detection
                                    img, result = self.detection(img)
                       
                                    # Add results to class count dict
                                    results[image_name] = result

                                    # Copy label output in output folder
                                    if(self.save_bb_results):
                                        self.generate_labels(image_name, checkpoint_dir)

                                    # Save image
                                    if(self.save_pred_images):
                                        self.save_image(img, image_name, checkpoint_dir)

                                # Save results in a text file
                                if(self.export_results):
                                    self.save_result_file(results, checkpoint_dir)

                        # If there is no checkpoint file in the weights folder
                        else:
                            print(f"The weights folder in {dir_name} does not contain .pt files!")
                    else:
                        # If there is no weights folder in the trainings folder
                        print(f"The folder {dir_name} does not contain a folder called weights!")
            # If the folder does not contain any training folders
            else:
                print(f"The specified folder {self.pth_batch_test} does not contain any training folders!")
        # If there are no images in the prediction folder      
        else:
            print(f'No images for prediction were found in folder {self.pth_predictions}!')
        

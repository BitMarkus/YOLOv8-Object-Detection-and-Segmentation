############################
# AutomaticImage Splitting #
############################

import os
from os import listdir
from os.path import isfile, join
import random
import shutil
from pathlib import Path
from random import randint
# Own modules
from settings import setting

class Img_Split():

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self):

        # Split settings
        self.img_extension = setting["split_img_extension"]
        self.val_split = setting["split_val_split"]
        self.test_split = setting["split_test_split"]
        self.num_bg_img = setting["split_num_bg_img"]

        # Paths
        self.num_bg_img = setting["split_num_bg_img"]
        # Input paths
        self.pth_splitin_data_img = setting["pth_splitin_data_img"]
        self.pth_splitin_data_label = setting["pth_splitin_data_label"]
        self.pth_splitin_bg_img = setting["pth_splitin_bg_img"]
        self.pth_splitin_bg_label = setting["pth_splitin_bg_label"]
        # Output paths
        self.pth_splitout = setting["pth_splitout"]
        self.pth_splitout_images_train = setting["pth_splitout_images_train"]
        self.pth_splitout_images_val = setting["pth_splitout_images_val"]
        self.pth_splitout_images_test = setting["pth_splitout_images_test"]
        self.pth_splitout_labels_train = setting["pth_splitout_labels_train"]    
        self.pth_splitout_labels_val = setting["pth_splitout_labels_val"]
        self.pth_splitout_labels_test = setting["pth_splitout_labels_test"]            

    #############################################################################################################
    # METHODS:

    # Returns a list of image names (with extension) from a specific folder
    # Only includes images with the specified extension
    # Shuffles the list randomly using the seed parameter
    def load_img_list(self, pth, extension, seed):
        # Check if the folder exists within the predict folder
        if(os.path.isdir(pth)): 
            # Get a list of all files in the specified folder
            img_list = [f for f in listdir(pth) if isfile(join(pth, f)) and f.endswith(extension)]
            # Check if there are more images in the folder than images to predict
            if(len(img_list) == 0):
                print(f"No images with extension {extension} in folder!")  
                return False   
            else:
                # Shuffle list
                random.seed(seed)
                random.shuffle(img_list)
                return img_list
        else:
            print("Folder does not exist!")  
            return False 
        
    # Split a list according to a certain percentage (0.0 - 1.0) TWICE
    # Returns all three truncated lists
    # No random split necessary as the list is already shuffled
    def split_list_perc(self, list, perc_1, perc_2):
        num_img = len(list)
        if(num_img > 0):
            # Calculate split indices of list
            split_id_1 = round(num_img * perc_1)
            split_id_2 = round(num_img * perc_2) + split_id_1
            # Split list in three parts
            list_1 = list[:split_id_1]
            list_2 = list[split_id_1:split_id_2]
            list_3 = list[split_id_2:]
            return list_1, list_2, list_3
        else:
            print("List is empty!")  
            return [], [], []   

    # Split a list at a certain index in two lists
    # Returns only the first list, which is num entries long
    # No random split necessary as the list was already shuffled
    def split_list_idx(self, list, num):
        num_img = len(list)
        if(num_img > 0):
            if(num <= num_img):
                # Split list
                list_1 = list[:num]
                # list_2 = list[num:]
                return list_1
            else:
                print("Not enough list entries for this split!")  
                return []             
        else:
            print("List is empty!")  
            return []    

    # Copies images, which are specified in parameter list, 
    # from one folder (sorc_pth) to another (dest_pth)
    def copy_img(self, list, sorc_pth, dest_pth): 
        if(os.path.isdir(sorc_pth) and os.path.isdir(dest_pth)):  
            for img_name in list:
                src = sorc_pth + img_name
                dst = dest_pth + img_name
                shutil.copy(src, dst)
            return True
        else:
            print("Source and/or destination folder does not exist!")  
            return False 

    # Copies label txt files, which are specified in parameter list, 
    # from one folder (sorc_pth) to another (dest_pth)
    def copy_label(self, list, sorc_pth, dest_pth):
        if(os.path.isdir(sorc_pth) and os.path.isdir(dest_pth)):  
            for img_file in list:
                # Remove extension
                file_name, ext = os.path.splitext(img_file)
                # Replace extension
                label_file = file_name + '.txt'
                # print(label_file)
                src = sorc_pth + label_file
                dst = dest_pth + label_file
                shutil.copy(src, dst)
            return True
        else:
            print("Source and/or destination folder does not exist!")  
            return False  

    # Creates the folder structure for one generated dataset
    # The folderg go into the specified folder/path ds_pth
    def create_dataset_folders(self, ds_pth):
        # Check if path already exists
        if(os.path.exists(ds_pth)):
            print("Dataset cannot be created. A folder with the same seed alredy exists!") 
        else:
            Path(ds_pth).mkdir(parents=True)
            Path(ds_pth + self.pth_splitout_images_train).mkdir(parents=True)
            Path(ds_pth + self.pth_splitout_images_val).mkdir(parents=True)
            Path(ds_pth + self.pth_splitout_images_test).mkdir(parents=True)
            Path(ds_pth + self.pth_splitout_labels_train).mkdir(parents=True)
            Path(ds_pth + self.pth_splitout_labels_val).mkdir(parents=True)
            Path(ds_pth + self.pth_splitout_labels_test).mkdir(parents=True)           

    #############################################################################################################
    # CALL:

    def __call__(self):

        # Generate random shuffle seeds for as much datasets as requested
        num_ds = input("Number of datasets to generate: ")
        seed_list = []
        for _ in range(int(num_ds)):
            seed_list.append(randint(1, 1000))
        # print(seed_list)

        print("Generate datasets. Please wait...")
        # Iterate over shuffle seeds
        for seed in seed_list: 

            # Generate directory with sub directories for dataset
            # Name of the dataset is the random shuffle seed
            ds_pth = f"{self.pth_splitout}dataset_{seed}/"
            print(f"> Generate dataset {seed}...")

            # Create dataset folder
            self.create_dataset_folders(ds_pth)

            # Load a list of all image names in the data folder
            img_list = self.load_img_list(self.pth_splitin_data_img, self.img_extension, seed)

            # Choose randomly x% of all images and move them to another list (validation images)
            val_list, test_list, train_list = self.split_list_perc(img_list, self.val_split, self.test_split)   
            # print(train_list)     
            # print(val_list)      
            # print(test_list)   

            # Make a list of all image names in the background folder and choose randomly x images
            # https://github.com/ultralytics/yolov5/issues/2844
            # An image without a label and an image with an empty label are both considered background images
            bg_list = self.load_img_list(self.pth_splitin_bg_img, self.img_extension, seed)
            bg_list = self.split_list_idx(bg_list, self.num_bg_img)
            # print(bg_list)

            # Copy images and labels from pos and neg lists to their respective train folders
            # Background images are only copied to the training images 
            # and not to the validation and test images in this iteration
            self.copy_img(train_list, self.pth_splitin_data_img, ds_pth+self.pth_splitout_images_train)
            self.copy_img(bg_list, self.pth_splitin_bg_img, ds_pth+self.pth_splitout_images_train)
            self.copy_label(train_list, self.pth_splitin_data_label, ds_pth+self.pth_splitout_labels_train)
            self.copy_label(bg_list, self.pth_splitin_bg_label, ds_pth+self.pth_splitout_labels_train)

            # Copy images and labels from validation list to their respective validation folders
            self.copy_img(val_list, self.pth_splitin_data_img, ds_pth+self.pth_splitout_images_val)
            self.copy_label(val_list, self.pth_splitin_data_label, ds_pth+self.pth_splitout_labels_val)
        
            # Copy images and labels from test list to their respective test folders
            self.copy_img(test_list, self.pth_splitin_data_img, ds_pth+self.pth_splitout_images_test)
            self.copy_label(test_list, self.pth_splitin_data_label, ds_pth+self.pth_splitout_labels_test)

            print(f"Dataset {seed} was saved in folder {ds_pth}.")

        print("Datasets were successfully created.")
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
        # Number of datasets to create
        self.num_datasets = setting["split_num_datasets"]
        # Extension of training images
        self.img_extension = setting["split_img_extension"]
        # Validation split
        self.is_val_split = setting["split_is_val_split"]
        if(self.is_val_split):
            self.val_split = setting["split_val_split"]
        else:
            self.val_split = 0.0
        # Test split
        self.is_test_split = setting["split_is_test_split"]
        if(self.is_test_split):
            self.test_split = setting["split_test_split"]
        else:
            self.test_split = 0.0
        # Background images
        self.use_bg_img = setting["split_use_bg_img"]
        self.use_bg_img_for_train = setting["split_use_bg_img_for_train"]
        self.use_bg_img_for_val = setting["split_use_bg_img_for_val"]
        self.use_bg_img_for_test = setting["split_use_bg_img_for_test"]
        self.bg_split = setting["split_bg_split"]

        # Paths
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
            Path(ds_pth + self.pth_splitout_labels_train).mkdir(parents=True)
            if(self.is_val_split):
                Path(ds_pth + self.pth_splitout_images_val).mkdir(parents=True)
                Path(ds_pth + self.pth_splitout_labels_val).mkdir(parents=True)
            if(self.is_test_split):
                Path(ds_pth + self.pth_splitout_images_test).mkdir(parents=True)
                Path(ds_pth + self.pth_splitout_labels_test).mkdir(parents=True)           

    #############################################################################################################
    # CALL:

    def __call__(self):

        # Generate random shuffle seeds (1-1000) for max 999 datasets
        seed_list = [randint(1, 1000) for s in range(self.num_datasets)]

        print(f"Generate {self.num_datasets} datasets. Please wait...")

        # Iterate over shuffle seeds
        for seed in seed_list: 

            # Load a list of all image names in the data folder and shuffle it
            img_list = self.load_img_list(self.pth_splitin_data_img, self.img_extension, seed)
            if(img_list):

                print(f"\n> Generate dataset {seed}...")

                # Generate directory with sub directories for dataset
                # Name of the dataset is the random shuffle seed
                ds_pth = f"{self.pth_splitout}dataset_{seed}/"
                # Create dataset folder
                self.create_dataset_folders(ds_pth)

                # Choose randomly x% of all images and move them to another list (validation images)
                val_list, test_list, train_list = self.split_list_perc(img_list, self.val_split, self.test_split)   
                print(f"Number of images: train={len(train_list)}, val={len(val_list)}, test={len(test_list)}") 
                
                # Make a list of all image names in the background folder and choose randomly x images
                # https://github.com/ultralytics/yolov5/issues/2844
                # An image without a label and an image with an empty label are both considered background images
                # Here, background images have empty label txt files
                if(self.use_bg_img):

                    # Calculate number of background images for each list
                    # Training images
                    num_bg_train_img = 0
                    if(self.use_bg_img_for_train):
                        num_train_img = len(train_list)
                        if(num_train_img > 0):
                            num_bg_train_img = round(num_train_img * self.bg_split)
                    # Validation images
                    num_bg_val_img = 0
                    if(self.is_val_split and self.use_bg_img_for_val):
                        num_val_img = len(val_list)
                        if(num_val_img > 0):
                            num_bg_val_img = round(num_val_img * self.bg_split)                            
                    # Test images
                    num_bg_test_img = 0
                    if(self.is_test_split and self.use_bg_img_for_test):
                        num_test_img = len(test_list)
                        if(num_test_img > 0):
                            num_bg_test_img = round(num_test_img * self.bg_split)

                    # Calculate the total amount of background images
                    num_bg_all_img =  num_bg_train_img + num_bg_val_img + num_bg_test_img   
                    print(f"Number of bg images: train={num_bg_train_img}, val={num_bg_val_img}, test={num_bg_test_img}")                      

                    # Load all background images into a list
                    bg_list = self.load_img_list(self.pth_splitin_bg_img, self.img_extension, seed)
                    if(bg_list):
                        # Split the list
                        train_bg_list = bg_list[:num_bg_train_img]
                        val_bg_list = bg_list[num_bg_train_img:num_bg_train_img+num_bg_val_img]
                        test_bg_list = bg_list[num_bg_train_img+num_bg_val_img:num_bg_all_img]
                        num_bg_list = len(bg_list)
                    else:
                        train_bg_list = False
                        val_bg_list = False
                        test_bg_list = False
                        num_bg_list = 0

                    if(num_bg_all_img > num_bg_list):
                        print("Warning: Not enough background images! No background images will be added.")
                        train_bg_list = False
                        val_bg_list = False
                        test_bg_list = False
                    
                    # print(bg_list_train)
                    # print(bg_list_val)
                    # print(bg_list_test)

                # Copy images and labels from data and background lists to their respective output folders
                # Training images and labels
                self.copy_img(train_list, self.pth_splitin_data_img, ds_pth+self.pth_splitout_images_train)
                self.copy_label(train_list, self.pth_splitin_data_label, ds_pth+self.pth_splitout_labels_train)
                # Background
                if(self.use_bg_img and self.use_bg_img_for_train and train_bg_list):
                    self.copy_img(train_bg_list, self.pth_splitin_bg_img, ds_pth+self.pth_splitout_images_train)
                    self.copy_label(train_bg_list, self.pth_splitin_bg_label, ds_pth+self.pth_splitout_labels_train)

                # Validation images and labels
                if(self.is_val_split):
                    self.copy_img(val_list, self.pth_splitin_data_img, ds_pth+self.pth_splitout_images_val)
                    self.copy_label(val_list, self.pth_splitin_data_label, ds_pth+self.pth_splitout_labels_val)
                    # Background
                    if(self.use_bg_img and self.use_bg_img_for_val and val_bg_list):
                        self.copy_img(val_bg_list, self.pth_splitin_bg_img, ds_pth+self.pth_splitout_images_val)
                        self.copy_label(val_bg_list, self.pth_splitin_bg_label, ds_pth+self.pth_splitout_labels_val)

                # Test images and labels
                if(self.is_test_split):
                    self.copy_img(test_list, self.pth_splitin_data_img, ds_pth+self.pth_splitout_images_test)
                    self.copy_label(test_list, self.pth_splitin_data_label, ds_pth+self.pth_splitout_labels_test)
                    # Background
                    if(self.use_bg_img and self.use_bg_img_for_test and test_bg_list):
                        self.copy_img(test_bg_list, self.pth_splitin_bg_img, ds_pth+self.pth_splitout_images_test)
                        self.copy_label(test_bg_list, self.pth_splitin_bg_label, ds_pth+self.pth_splitout_labels_test)

                print(f"Dataset {seed} was saved in folder {ds_pth}.")

            # If there are no images at this path or if the path does not exist
            else:
                print(f"Path {self.pth_splitin_data_img} does not exist or there are no images inside!")

        print("\nDatasets were successfully created.")

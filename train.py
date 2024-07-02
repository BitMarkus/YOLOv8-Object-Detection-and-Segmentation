####################
# Network Training #
####################

from pathlib import Path
import os
# Own modules
from ultralytics import YOLO
from settings import setting

class Train:

    #############################################################################################################
    # CONSTRUCTOR:
    
    def __init__(self):
        # Path to dataset info
        self.pth_dataset_info = f'{setting["pth_dataset_info"]}{setting["train_dataset_info_name"]}'
        # Path to training output
        self.pth_training_output = setting["pth_output"]
        # Selection of model for training
        # Either an empty model or a pretrained model (yolov8 models) can be used for training
        # Model name strings
        self.model_prefix = "yolov8"
        self.empty_model_extension = ".yaml"
        self.pretrained_model_extension = ".pt"
        # Set model for training
        self.use_pretrained_model = setting["train_use_pretrained_model"]
        self.model_name = f'{self.model_prefix}{setting["train_model_size"]}'
        if(self.use_pretrained_model):
            self.model_file = f'{self.model_name}{self.pretrained_model_extension}'
        else:
            self.model_file = f'{self.model_name}{self.empty_model_extension}'
        self.model = YOLO(f'{setting["pth_yolo_models"]}{self.model_file}')
        # Network hyperparameters
        self.num_epochs = setting["train_num_epochs"]
        self.batch_size = setting["train_batch_size"]
        self.patience = setting["train_patience"]
        # Image size and shape for training
        self.train_img_size = setting["train_img_size"]
        self.train_rectangular_img = setting["train_rectangular_img"]
        # Train validation
        self.do_validation = setting["train_validation"]
        self.verbose = setting["train_verbose"]
        self.seed = setting["train_seed"]
        self.save_plots = setting["train_save_plots"]
        # Save checkpoints each x epochs
        self.chckpt_save_period = setting["train_chckpt_save_period"]
        # Emphasize box or class accuracy
        self.box_emph = setting["train_box_emph"]
        self.class_emph = setting["train_class_emph"]
        # Segmentation training
        self.overlap_mask = setting["train_overlap_mask"]
        self.mask_ratio = setting["train_mask_ratio"]
        # Dataset parameters
        self.train_classes = setting["train_classes"]
        self.pth_training_images = setting["pth_training_images"]
        self.pth_validation_images = setting["pth_validation_images"]
        # Augmentation parameters
        self.aug_hsv_h = setting["aug_hsv_h"]
        self.aug_hsv_s = setting["aug_hsv_s"]
        self.aug_hsv_v = setting["aug_hsv_v"]
        self.aug_degrees = setting["aug_degrees"]
        self.aug_translate = setting["aug_translate"]
        self.aug_scale = setting["aug_scale"]
        self.aug_shear = setting["aug_shear"]
        self.aug_perspective = setting["aug_perspective"]
        self.aug_flipud = setting["aug_flipud"]
        self.aug_fliplr = setting["aug_fliplr"]
        self.aug_mosaic = setting["aug_mosaic"]
        self.aug_mixup = setting["aug_mixup"]
        self.aug_copy_paste = setting["aug_copy_paste"]

    #############################################################################################################
    # METHODS:

    # Dynamically creates the dataset.yaml file, which is necessary for training
    def generate_dataset_info(self):
        # get the programs root directory path
        home = os.getcwd()
        # Create a new dataset.yaml file 
        # This mode opens the file for writing only. The data in existing files are modified and overwritten
        # If the file does not already exist in the folder, a new one gets created
        f = open(self.pth_dataset_info, "w")
        # Write home and image directories
        f.write(f'path: {home}\n') 
        f.write(f'train: {self.pth_training_images}\n')
        f.write(f'val: {self.pth_validation_images}\n\n')
        # Write classes
        f.write('names:\n') 
        for key in self.train_classes:
            f.write(f'  {key}: {self.train_classes[key]}\n') 
        f.write('\n') 
        # Write augmentation list
        f.write('augmentations:\n') 
        f.write(f'  hsv_h: {self.aug_hsv_h}\n')
        f.write(f'  hsv_s: {self.aug_hsv_s}\n')
        f.write(f'  hsv_v: {self.aug_hsv_v}\n')
        f.write(f'  degrees: {self.aug_degrees}\n')
        f.write(f'  translate: {self.aug_translate}\n')
        f.write(f'  scale: {self.aug_scale}\n')
        f.write(f'  shear: {self.aug_shear}\n')
        f.write(f'  perspective: {self.aug_perspective}\n')
        f.write(f'  flipud: {self.aug_flipud}\n')
        f.write(f'  fliplr: {self.aug_fliplr}\n')
        f.write(f'  mosaic: {self.aug_mosaic}\n')
        f.write(f'  mixup: {self.aug_mixup}\n')
        f.write(f'  copy_paste: {self.aug_copy_paste}')
        f.close()

    # Deletes the dataset.yaml file
    def delete_dataset_info(self):
        if os.path.exists(self.pth_dataset_info):
            os.remove(self.pth_dataset_info)

    #############################################################################################################
    # CALL:

    def __call__(self):

        # Generate dataset.yaml file
        self.generate_dataset_info()

        # train the model 
        # Train parameters: https://docs.ultralytics.com/usage/cfg/#train-settings   
        self.model.train(
            model=self.model,
            data=self.pth_dataset_info, 
            pretrained=self.use_pretrained_model,
            imgsz=self.train_img_size,
            epochs=self.num_epochs,
            batch=self.batch_size,
            patience=self.patience,
            val=self.do_validation,
            verbose=self.verbose,
            seed=self.seed,
            plots=self.save_plots,
            project=self.pth_training_output,
            rect=self.train_rectangular_img,
            save_period=self.chckpt_save_period,
            box=self.box_emph,
            cls=self.class_emph,
            overlap_mask=self.overlap_mask,
            mask_ratio=self.mask_ratio,
            save=True,
        )   

        # Delete dataset.yaml file after training
        # self.delete_dataset_info() 

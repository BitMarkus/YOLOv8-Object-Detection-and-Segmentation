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

####################
# Network Training #
####################

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
        # Training hyperparameters
        self.hyp_optimizer = setting["train_optimizer"]
        self.hyp_cos_lr = setting["train_cos_lr"]
        self.hyp_lr0 = setting["train_lr0"]
        self.hyp_lrf = setting["train_lrf"]
        self.hyp_momentum = setting["train_momentum"]
        self.hyp_weight_decay = setting["train_weight_decay"]
        self.hyp_warmup_epochs = setting["train_warmup_epochs"]
        self.hyp_warmup_momentum = setting["train_warmup_momentum"]
        self.hyp_warmup_bias_lr = setting["train_warmup_bias_lr"]
        self.hyp_dropout = setting["train_dropout"]        

    #############################################################################################################
    # METHODS:

    # Dynamically creates the dataset.yaml file, which is necessary for training
    def generate_dataset_info(self):
        # get the programs root directory path
        home = os.getcwd()
        # Create a new dataset.yaml file 
        # This mode opens the file for writing only. The data in existing files are modified and overwritten
        # If the file does not already exist in the folder, a new one gets created
        with open(self.pth_dataset_info, "w") as f:
            # Write home and image directories
            f.write(f'path: {home}\n') 
            # Replace \ by / in the string??? -> Works at least in Windows as expected
            f.write(f'train: {self.pth_training_images}\n')
            f.write(f'val: {self.pth_validation_images}\n\n')
            # Write classes
            f.write('names:\n') 
            for key in self.train_classes:
                f.write(f'  {key}: {self.train_classes[key]}\n') 
                
            """
            # Augentations in the config.yaml file are no longer accepted
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
            f.write('\n\n') 
            # Write training hyperparameters list
            f.write('hyperparameters:\n') 
            f.write(f"  optimizer: '{self.hyp_optimizer}'\n")
            f.write(f'  cos_lr: {self.hyp_cos_lr}\n')
            f.write(f'  lr0: {self.hyp_lr0}\n')
            f.write(f'  lrf: {self.hyp_lrf}\n')
            f.write(f'  momentum: {self.hyp_momentum}\n')
            f.write(f'  weight_decay: {self.hyp_weight_decay}\n')
            f.write(f'  warmup_epochs: {self.hyp_warmup_epochs}\n')
            f.write(f'  warmup_momentum: {self.hyp_warmup_momentum}\n')
            f.write(f'  warmup_bias_lr: {self.hyp_warmup_bias_lr}\n')
            f.write(f'  dropout: {self.hyp_dropout}\n')
            """

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
            # Training hyperparameters
            epochs=self.num_epochs,
            batch=self.batch_size,
            optimizer=self.hyp_optimizer,
            cos_lr=self.hyp_cos_lr,
            lr0=self.hyp_lr0,
            lrf=self.hyp_lrf,
            momentum=self.hyp_momentum,
            weight_decay=self.hyp_weight_decay,
            warmup_epochs=self.hyp_warmup_epochs,
            warmup_momentum=self.hyp_warmup_momentum,
            warmup_bias_lr=self.hyp_warmup_bias_lr,
            dropout=self.hyp_dropout,
            # Augmentation parameters
            hsv_h=self.aug_hsv_h,
            hsv_s=self.aug_hsv_s,
            hsv_v=self.aug_hsv_v,
            degrees=self.aug_degrees,
            translate=self.aug_translate,
            scale=self.aug_scale,
            shear=self.aug_shear,
            perspective=self.aug_perspective,
            flipud=self.aug_flipud,
            fliplr=self.aug_fliplr,
            mosaic=self.aug_mosaic,
            mixup=self.aug_mixup,
            copy_paste=self.aug_copy_paste,
        )

        # Delete dataset.yaml file after training
        # self.delete_dataset_info() 


####################
# Program settings #
####################

setting = {

    ############
    # TRAINING #
    ############

    # Determines whether to start training from a pretrained model. 
    # Can be a boolean value or a string path to a specific model from which to load weights
    "train_use_pretrained_model": False,
    # Size of the pretrained model (n, s, m, l, x)
    "train_model_size": "m",
    # Name of the necessary train info located in folder /train
    "train_dataset_info_name": "config.yaml",
    # Dict of classes for automatic generation of dataset.yaml file
    "train_classes": {0: 'no cilium', 1: 'cilium', 2: 'nucleus', 3: 'mitosis'},
    # "train_classes": {0: 'PC'},

    # Training parameters
    # https://docs.ultralytics.com/usage/cfg/#train-settings
    # Total number of training epochs. Each epoch represents a full pass over the entire dataset
    "train_num_epochs": 5000,
    # Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory 
    # utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70)
    "train_batch_size": 4, # 2- for 2048 and x, 6 for 1024 and x, 8+ for 1024 and m, 15+ for 1024 and n
    # Number of epochs to wait without improvement in validation metrics before early stopping the training.
    # 0 for no early stopping
    "train_patience": 0,
    # Enables validation during training, allowing for periodic evaluation of model performance on a separate dataset
    "train_validation": True,
    # Target image size for training. All images are resized to this dimension before being fed into the model
    "train_img_size": 2048,     # 2048, 1024
    # Enables rectangular training, optimizing batch composition for minimal padding.
    # TRAINING WARNING: 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False 
    # Eventually remove this from training
    "train_rectangular_img": False,
    # Enables verbose output during training, providing detailed logs and progress updates
    "train_verbose": True,
    # Sets the random seed for training, ensuring reproducibility of results across runs with the same configurations
    "train_seed": 111,
    # Generates and saves plots of training and validation metrics, as well as prediction examples, 
    # providing visual insights into model performance and learning progression
    "train_save_plots": True,
    # Frequency of saving model checkpoints, specified in epochs. A value of -1 disables this feature
    "train_chckpt_save_period": 500,
    # Weight of the box loss component in the loss function, influencing how much emphasis is 
    # placed on accurately predicting bounding box coordinates
    # Standart: 7.5
    "train_box_emph": 7.5,
    # Weight of the classification loss in the total loss function, affecting the importance 
    # of correct class prediction relative to other components
    # Standart: 0.5
    "train_class_emph": 0.5,
    # Segmentation training:
    # Determines whether segmentation masks should overlap during training, applicable in instance segmentation tasks.
    "train_overlap_mask": True,
    # Downsample ratio for segmentation masks, affecting the resolution of masks used during training.
    # Standart: 4
    "train_mask_ratio": 4,

    # Training hyperparameters
    # Choice of optimizer for training. Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., 
    # or 'auto' for automatic selection based on model configuration
    # Default = auto
    "train_optimizer": 'auto',
    # Utilizes a cosine learning rate scheduler, adjusting the learning rate following a cosine curve over epochs
    # Default = False
    "train_cos_lr": False,
    # Initial learning rate (i.e. SGD=1E-2, Adam=1E-3) . Adjusting this value is crucial for the 
    # optimization process, influencing how rapidly model weights are updated
    # Default = 0.01
    "train_lr0": 0.01,
    # Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with 
    # schedulers to adjust the learning rate over time
    # Default = 0.01
    "train_lrf": 0.01,
    # Momentum factor for SGD or beta1 for Adam optimizers, influencing the incorporation of past gradients in the current update
    # Default = 0.937
    "train_momentum": 0.937,
    # L2 regularization term, penalizing large weights to prevent overfitting
    # Default = 0.0005
    "train_weight_decay": 0.0005,
    # Number of epochs for learning rate warmup, gradually increasing the learning rate from a low 
    # value to the initial learning rate to stabilize training early on
    # Default = 3.0
    "train_warmup_epochs": 3.0,
    # Initial momentum for warmup phase, gradually adjusting to the set momentum over the warmup period
    # Default = 0.8
    "train_warmup_momentum": 0.8,
    # Learning rate for bias parameters during the warmup phase, helping stabilize model training in the initial epochs
    # Default = 0.1
    "train_warmup_bias_lr": 0.1,
    # Dropout rate for regularization in classification tasks, preventing overfitting by randomly omitting units during training
    # Default = 0.0
    "train_dropout": 0.0,

    # Data augmentation
    # https://github.com/orgs/ultralytics/discussions/4142
    # https://docs.ultralytics.com/usage/cfg/#augmentation-settings
    # Adjust the hue, saturation, and value of the image colors to introduce color variability
    # Adjusts the hue of the image by a fraction of the color wheel, introducing color variability. 
    # Helps the model generalize across different lighting conditions.
    "aug_hsv_h": 0.0,  # 0.015
    # Alters the saturation of the image by a fraction, affecting the intensity of colors. 
    # Useful for simulating different environmental conditions.
    "aug_hsv_s": 0.0,   # 0.7
    # Modifies the value (brightness) of the image by a fraction, helping the model to perform 
    # well under various lighting conditions.
    "aug_hsv_v": 0.0,   # 0.4
    # Rotate the image by a certain degree to simulate different orientations
    "aug_degrees": 0.0, 
    # Translate the image horizontally and vertically to simulate different positions
    "aug_translate": 0.1, 
    # Scale the image to simulate different sizes of objects
    "aug_scale": 0.5,
    # Shear the image to simulate perspective changes
    "aug_shear": 0.0,
    # Adjust the perspective of the image
    "aug_perspective": 0.0,
    # Flip the image upside down or left to right
    "aug_flipud": 0.5,  # flip up and down
    "aug_fliplr": 0.5,  # flip left and right
    # Combine four training images into one to simulate different contexts
    "aug_mosaic": 1.0,
    # Overlay two images to create a single composite image
    "aug_mixup": 0.0,
    # For segmentation tasks, copy objects from one image and paste them onto another
    "aug_copy_paste": 0.0,

    ####################
    # OBJECT DETECTION #
    ####################

    # Set to True if a pretrained yolov8 model is used
    # Set to False for a custom/self trained model
    "od_use_pretrained_model": False,
    # Set to True, if a segmentation model is used (ends with -seg.pt)
    "od_use_segmentation_model": False,
    # Size of the pretrained model (n, s, m, l, x)
    "od_pretrained_model_size": "m",
    # Name of custom model, in case 'od_use_pretrained_model' is set to False
    "od_custom_model_name": "cilia_3000e_ds9_2_m_2048px_best.pt",

    # https://docs.ultralytics.com/usage/cfg/#predict-settings
    # Filters predictions to a set of class IDs. Only detections belonging to the specified classes will be returned
    # empy list = all classes
    "od_class_selection": [],
    # Maximum number of detections allowed per image. Limits the total number of objects the model
    # can detect in a single inference, preventing excessive outputs in dense scenes (default 300)
    "od_max_detections": 500,
    # Intersection Over Union (IoU) threshold: 
    # Lower values result in fewer detections by eliminating overlapping boxes, 
    # useful for reducing duplicates (default 0.7)
    "od_iou": 0.6,
    # Sets the minimum confidence threshold for detections. Objects detected with confidence below 
    # this threshold will be disregarded
    "od_min_conf": 0.4,

    # Settings for bounding box annotations
    # Show labels AND confidence on top of the bounding boxes
    "od_show_labels": False,
    # Show bounding boxes
    "od_show_bbox": True,
    # Bounding box line thickness
    "od_bbox_line_thickness": 1,
    # Labels and confidence text thickness
    "od_bbox_text_thickness": 1,
    # Labels and confidence text size
    "od_bbox_text_scale": 0.4,

    ### DETECTION FOR IMAGES ###
    # Show predicted images after prediction
    # Press any key to close image
    "od_show_predicted_images": False, 
    # Set to true if predicted images are supposed to be saved 
    "od_save_predicted_images": False, 
    # Defines the image size for inference for images (w, h)
    "od_inf_size_img": (2048, 2048),    # (1024, 1024), (2048, 2048)
    # Set to true if images for prediction are rectangeled,
    # Set to false if images are squared
    "od_rectangular_img": False,
    # If this is set to true, a text file with the detection 
    # results will be saved in the output folder
    "od_export_results": True,
    # Name of the result file
    "od_export_file_name": 'results.txt',
    # Save bounding box results from predictions as txt file
    "od_save_bb_results": True,

    ### DETECTION FOR VIDEOS ###
    # Show predicted output video
    # Press 'Esc' to close video window, press 's'
    "od_show_output_video": True,
    # Set to true if predicted video is supposed to be saved
    "od_save_output_video": False, 
    # Name of the output video file (if saved)
    # Currently only one video can be recorded. A second record will overwrite the old one due to the same name!
    "od_output_video_name": "object_detection",
    # Webcam index
    "od_video_source": 0,  
    # Set width, hight and fps of webcam image for display
    "od_frame_width": 1920, # 1920
    "od_frame_height": 1080, # 1080
    "od_frame_fps": 25,
    # File name for saved video frames
    "od_saved_video_frame_prefix": "video_frame_",
    # File extension for saved frames
    "od_saved_video_frame_extension": ".png",
    # Defines the image size for inference for videos (w, h)
    # Results are as expected when width and height is changed:
    # https://github.com/ultralytics/ultralytics/issues/3955
    "od_inf_size_vid": (768, 1344),
    # Set to true if video for prediction is rectangeled,
    # Set to false if video is squared    
    "od_rectangular_video": False,

    ### DETECTION FROM SCREEN ###
    # Monitor selection for multi monitor setup
    "od_monitor_number": 2,
    # Offset of capture region from the left upper corner of the monitor
    "od_capture_roi_offset_xy": (28, 208),
    # Width and height of the capture region
    "od_capture_roi_size_wh": (1336, 752),
    # FPS for capture video recording
    "od_capture_roi_record_fps": 12,

    ### RESULT WINDOW (VIDEO AND IMAGES) ###
    # Result window title
    "window_results_title": "YOLO Detection: Results",
    # Result window position from the left upper corner in px
    "window_results_x_pos": 100,
    "window_results_y_pos": 100,

    #######################
    # OBJECT SEGMENTATION #
    #######################

    # Show segmentation masks
    "os_show_mask": True,
    # Show outlines of segmented areas
    "os_show_outline": True,
    # Thickness of outlines
    "os_outline_thickness": 1, # DOES NOTHING!?!
    # Color of outlines
    "os_outline_color": (255, 0, 0), # blue

    ###########################
    # OBJECT TRACKING (VIDEO) #
    ###########################

    # https://docs.ultralytics.com/guides/object-counting/#argument-objectcounter  
    # Region points for object counting
    "oc_roi_start": (10, 10),
    "oc_roi_size": (1316, 732), # (1900, 1060)
    # Sets the minimum confidence threshold for counting
    "oc_min_conf": 0.4,
    # List of classes to count, empy list = all classes
    "oc_class_selection": [],
    # Settings for bounding box annotations
    # Counter box color
    # In OpenCV BGR format -> red
    "oc_box_color": (0, 0, 255),
    # Counter box line thickness
    "oc_box_line_thickness": 1,
    # Bounding boxes line thickness
    "oc_bboxes_line_thickness": 1,
    # In/out display text color
    # In OpenCV BGR format -> white
    "oc_text_color": (255, 255, 255),
    # In/out display background color
    # In OpenCV BGR format -> black
    "oc_bg_color": (0, 0, 0),
    # Set to true if object tracks are suppose to be shown
    "oc_draw_tracks": True,
    # Line thickness of object tracks
    "oc_track_thickness": 2,
    # Show in/out counts
    "oc_view_in_counts": True,
    "oc_view_out_counts": True,

    #################
    # CLASS COUNTER #
    #################

    # Show class counter
    "od_show_class_counter": True,
    # x,y-coordinates of the class cunter from the left upper corner in px
    "od_counter_offset": (10, 40), 
    # Line height, depends on font size (scale)
    "od_counter_line_height": 25,
    # Font size (determines line height)
    "od_counter_scale": 0.7,
    # Class counter text color
    # In OpenCV BGR format -> white    
    "od_counter_color": (255, 255, 255),
    # Class counter text thickness
    "od_counter_thickness": 1,
    
    ###############
    # FPS DISPLAY #
    ###############

    # Show FPS display
    "show_fps": True,
    # x,y-coordinates of the fps display from the left upper corner in px
    "fps_position": (10, 30), 
    # Font size
    "fps_font_scale": 0.7,
    # Font color
    # In OpenCV BGR format -> white 
    "fps_color": (255, 255, 255),
    # FPS display text thickness
    "fps_font_thickness": 1,

    ##################
    # IMAGE SPLITTER #
    ##################

    # Number of datasets to create
    "split_num_datasets": 2,
    # Extension of training images
    "split_img_extension": ".jpg",
    # Validation split
    "split_is_val_split": True,
    "split_val_split": 0.2, # 0.0 - 1.0
    # Test split
    "split_is_test_split": False,
    "split_test_split": 0.1, # 0.0 - 1.0
    # Background images
    "split_use_bg_img": True,
    "split_use_bg_img_for_train": True,
    "split_use_bg_img_for_val": True,
    "split_use_bg_img_for_test": True,
    "split_bg_split": 0.1, # 0.0 - 1.0

    #########
    # PATHS #
    #########

    "pth_models": "models/",
    "pth_yolo_models": "models/yolo_models/",
    "pth_custom_models": "models/custom_models/",
    "pth_output": "output/",
    "pth_predictions": "predictions/",
    "pth_video_frames": "output/vidshot/",
    "pth_dataset_info": "train/",
    "pth_training_images": "train/images/train/",
    "pth_validation_images": "train/images/val/",
    "pth_training_labels": "train/labels/train/",
    "pth_validation_labels": "train/labels/val/",

    # Path for batch prediction on training folders
    # This is NOT the path to checkpoints/models! 
    # It is the absolute path to any folder containing training folders
    "pth_batch_test": "F:\Arbeit",
    # "pth_batch_test": "D:/AI/Zilien Projekt Bilder/Trainings/",  

    # Pthts for image splitter     
    # Input paths
    "pth_splitin_data_img": "img_splitter/input/data/images/",
    "pth_splitin_data_label": "img_splitter/input/data/labels/",
    "pth_splitin_bg_img": "img_splitter/input/background/images/",
    "pth_splitin_bg_label": "img_splitter/input/background/labels/",
    # Output paths
    "pth_splitout": "img_splitter/output/",
    "pth_splitout_images_train": "images/train/",
    "pth_splitout_images_val": "images/val/",
    "pth_splitout_images_test": "images/test/",
    "pth_splitout_labels_train": "labels/train/",
    "pth_splitout_labels_val": "labels/val/",
    "pth_splitout_labels_test": "labels/test/",
}
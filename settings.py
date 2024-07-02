
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
    "train_model_size": "x",
    # Name of the necessary train info located in folder /train
    "train_dataset_info_name": "dataset.yaml",
    # Dict of classes for automatic generation of dataset.yaml file
    "train_classes": {0: 'zentrosom', 1: 'zilium',},

    # Training hyperparameters
    # https://docs.ultralytics.com/usage/cfg/#train-settings
    # Total number of training epochs. Each epoch represents a full pass over the entire dataset
    "train_num_epochs": 3000,
    # Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory 
    # utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70)
    "train_batch_size": 1,
    # Number of epochs to wait without improvement in validation metrics before early stopping the training.
    # 0 for no early stopping
    "train_patience": 0,
    # Enables validation during training, allowing for periodic evaluation of model performance on a separate dataset
    "train_validation": True,
    # Target image size for training. All images are resized to this dimension before being fed into the model
    "train_img_size": 1024,
    # Enables rectangular training, optimizing batch composition for minimal padding.
    # TRAINING WARNING: 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False 
    # Eventually remove this from training
    "train_rectangular_img": False,
    # Enables verbose output during training, providing detailed logs and progress updates
    "train_verbose": True,
    # Sets the random seed for training, ensuring reproducibility of results across runs with the same configurations
    "train_seed": 123,
    # Generates and saves plots of training and validation metrics, as well as prediction examples, 
    # providing visual insights into model performance and learning progression
    "train_save_plots": True,
    # Frequency of saving model checkpoints, specified in epochs. A value of -1 disables this feature
    "train_chckpt_save_period": 1000,
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

    # Data augmentation
    # Adjust the hue, saturation, and value of the image colors to introduce color variability
    "aug_hsv_h": 0.015,  # hue
    "aug_hsv_s": 0.7,   # saturation
    "aug_hsv_v": 0.4,   # value
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
    "aug_flipud": 0.0,  # flip up and down
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
    "od_pretrained_model_size": "x",
    # Name of custom model, in case 'od_use_pretrained_model' is set to False
    "od_custom_model_name": "cilia_x_5000e_pretrained.pt", # cilia_3000e_m_best.pt, nuclei_3000e_m_best.pt

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
    "od_iou": 0.7,
    # Sets the minimum confidence threshold for detections. Objects detected with confidence below 
    # this threshold will be disregarded
    "od_min_conf": 0.3,

    # Settings for bounding box annotations
    # Show labels AND confidence on top of the bounding boxes
    "od_show_labels": True,
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
    "od_save_predicted_images": True, 
    # Defines the image size for inference for images (w, h)
    "od_inf_size_img": (1024, 1024),
    # Set to true if images for prediction are rectangeled,
    # Set to false if images are squared
    "od_rectangular_img": True,

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
    "od_inf_size_vid": (1088, 1920),
    # Set to true if video for prediction is rectangeled,
    # Set to false if video is squared    
    "od_rectangular_video": False,

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
    "oc_roi_size": (1900, 1060),
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
    # "pth_training_labels": "train/labels/train/",
    # "pth_validation_labels": "train/labels/val/",   

}
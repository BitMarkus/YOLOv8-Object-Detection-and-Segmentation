
####################
# Program settings #
####################

setting = {
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

    ############
    # TRAINING #
    ############
    # https://docs.ultralytics.com/usage/cfg/#train-settings
    "train_use_pretrained_model": False,
    "train_model_size": "m",
    "train_model_prefix": "yolov8",
    "train_empty_model_extension": ".yaml",
    "train_pretrained_model_extension": ".pt",
    "train_dataset_info_name": "dataset.yaml",
    "train_num_epochs": 3000,
    "train_batch_size": 1,
    "train_patience": 0,
    "train_validation": True,
    "train_img_size": 1024,
    "train_verbose": True,
    "train_seed": 123,
    "train_save_plots": True,
    "train_close_mosaic": 0,

    ####################
    # OBJECT DETECTION #
    ####################
    # Use pretrained yolov8 model or custom/self trained model
    "od_use_pretrained_model": False,
    "od_use_segmentation_model": False,
    # Standard model
    "od_pretrained_model_size": "m",
    "od_model_prefix": "yolov8",
    "od_segmentation_model_postfix": "-seg",
    "od_pretrained_model_extension": ".pt",
    # Name of custom model
    "od_custom_model_name": "nuclei_3000e_m_best.pt", # cilia_3000e_m_best.pt, yolov8x-seg.pt
    # List of classes to detect, empy list = all classes
    "od_class_selection": [],
    # Defines the image size for inference
    "od_inf_img_size_width": 1024,      # Webcam: 1920
    "od_inf_img_size_height": 1024,     # Webcam: 1088
    # Limits the amount of detections on an image (default 300)
    "od_max_detections": 500,
    # Intersection Over Union (IoU) threshold: 
    # Lower values result in fewer detections by eliminating overlapping boxes, 
    # useful for reducing duplicates (default 0.7)
    "od_iou": 0.7,
    # Sets the minimum confidence threshold for detections
    "od_min_conf": 0.3,
    # Settings for bounding box annotations
    "od_show_labels": False,
    "od_show_bbox": True,
    "od_bbox_line_thickness": 1,
    "od_bbox_text_thickness": 1,
    "od_bbox_text_scale": 0.4,

    ### ON IMAGES ###
    "od_show_predicted_images": False,  
    "od_save_predicted_images": True,  

    ### ON VIDEOS ###
    "od_show_output_video": True,
    "od_save_output_video": False, 
    "od_output_video_name": "object_detection",
    "od_video_source": 0,  # Webcam index ot path to video
    # Set width and hight of webcam image for display
    "od_frame_width": 1920, # 1920
    "od_frame_height": 1080, # 1080
    "od_frame_fps": 15,
    # File name for saved frames
    "od_saved_video_frame_prefix": "video_frame_",
    "od_saved_video_frame_extension": ".png",

    ## FOR VIDEOS AND IMAGES ###
    # Result window
    "window_results_title": "YOLO Detection: Results",
    "window_results_x_pos": 100,
    "window_results_y_pos": 100,

    ####################################
    # OBJECT COUNTING/TRACKING (VIDEO) #
    ####################################
    # https://docs.ultralytics.com/guides/object-counting/#argument-objectcounter  
    # Region points for object counting
    "oc_roi_start": (10, 10),
    "oc_roi_size": (1900, 1060),
    # Sets the minimum confidence threshold for counting
    "oc_min_conf": 0.4,
    # List of classes to count, empy list = all classes
    "oc_class_selection": [],
    # Settings for bounding box annotations
    "oc_box_color": (0, 0, 255), # red
    "oc_box_line_thickness": 1,
    "oc_bboxes_line_thickness": 1,
    "oc_text_color": (255, 255, 255),
    "oc_bg_color": (0, 0, 0),
    "oc_draw_tracks": True,
    "oc_track_thickness": 2,
    "oc_view_in_counts": True,
    "oc_view_out_counts": True,

    #######################
    # OBJECT SEGMENTATION #
    #######################

    #################
    # CLASS COUNTER #
    #################
    "od_show_class_counter": True,
    "od_counter_offset": (10, 40), # x,y-coordinates from the left upper corner in px
    "od_counter_line_height": 25,
    "od_counter_scale": 0.7,
    "od_counter_color": (255, 255, 255), # white
    "od_counter_thickness": 1,
    
    ###############
    # FPS DISPLAY #
    ###############
    # Settings for FPS display
    "show_fps": True,
    "fps_position": (10, 30), # x,y-coordinates from the left upper corner in px
    "fps_font_scale": 0.7,
    "fps_color": (255, 255, 255), # white
    "fps_font_thickness": 1,
    
}
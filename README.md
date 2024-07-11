# YOLOv8 Object Detection
 
Object detection and segmentation on videos and images based on YOLOv8 (in python).

After cloning the github repository, the folder structure will automatically be created while the program is started for the first time. 

FOLDERS: 
- models/: Either self trained models (subfolder custom_models) or YOLOv8 models for detection or segmentation (subfolder yolo_models) must be placed in these folders.
- output/: All program output will go in this folder (training results, predictions on videos and images). It is possible to make shots of the camera image by pressing the button 's' while the live image is displayed. These images are saved in the subfolder vidshot/.
- predictions/: This folder is for videos and images for prediction. results are saved in folder output/.
- train/: Here, images and labels for training are located. Please check the official YOLOv8 documentation for preparing images for training. The config.yaml file, which is necessary for training, will be automatically created at the beginning of training and stored in the root of the train/ folder. Besides the paths to training/validation images and the classes it also contains hyperparameters for training and augmentation. All values can be set in the settings.pt file.

FUNCTIONS:
The software comes with a console menu to select different functions. All necessary settings for training, detection and display are located in the settings.py file. This is the only file which needs to be adjusted. The menu offers the following options:
- Train model: Trains of fine tunes a model with a custom dataset. All parameters for training can be set in the settings.py file.
- Object detection and object segmentation on images
- Object detection, segmentation, and tracking on videos (saved videos or live stream form a camera)
- Object detection, segmentation, and tracking on screen captures (defined region on a monitor)
- Videos and screen captures can be recorded by changing "od_save_output_video" in the settings.py file to True. Pressing the 's' key makes a snapshots of the running video or screen capture.

ENVIRONMENT:
- Runs on CUDA
- NVIDIA RTX 4090 or NVIDIA RTX A5000 (24 GB VRAM)
- Python:  3.8.19
- Pytorch: 2.3.1
- CUDA: 12.1
- Ultralytics YOLO: 8.2.30
- Supervision: 0.21.0
- Opencv: 4.10.0
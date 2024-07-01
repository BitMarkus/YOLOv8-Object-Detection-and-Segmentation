from ultralytics import YOLO
from settings import setting

class Train:
    
    def __init__(self):
        # Path to dataset info
        self.pth_dataset_info = f'{setting["pth_dataset_info"]}{setting["train_dataset_info_name"]}'
        # Path to training output --> DOES NOT WORK
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

    def __call__(self):
        # train the model 
        # # Train parameters: https://docs.ultralytics.com/usage/cfg/#train-settings   
        self.model.train(
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
        )      

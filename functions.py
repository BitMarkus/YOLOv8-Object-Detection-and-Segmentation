import torch
import ultralytics
import supervision
import sys
import cv2
from pathlib import Path
# Own modules
from settings import setting

#######################################
# Functions for validating user input #
#######################################

# Creates an input with prompt
# which checkes, if the input is empty
# If yes, the loop will continue until an input is entered
def input_empty(prompt):
    while(True):
        inp = input(prompt).strip()
        if(len(inp) == 0):
            print("No Input! Try again:")
        else:
            return inp   

# Creates an input with prompt
# which is checked, if the input is an integer number
# If not, the loop will continue until a valid number is entered
def input_int(prompt):
    while(True):
        nr = input(prompt)
        if not(check_int(nr)):
            print("Input is not an integer number! Try again:")
        else:
            return int(nr)  

# Creates an input for entering a prediction threshold
# If no value is entered, the standard threshold of 0.5 is used
def input_threshold(prompt):
    while(True):
        thr = input(prompt)
        if(thr == ""):
            print(f"Standard threshold value of 0.5 is used")
            return 0.5
        else:
            if not(check_nr(thr)):
                print("Input is not an number! Try again:")
            else:
                thr = float(thr)
                if(thr <= 0 or thr >= 1):
                    print("Threshold value must be > 0 and < 1! Try again:")
                else:
                    return thr    

####################################        
# Functions to check variable type #
####################################

# Check variable for number (int or float)
# Returns True if conversion was successful
# or False when the variable cannot be converted to a number
def check_nr(var):
    try:
        # Convert it into integer
        val = int(var)
        return True
    except ValueError:
        try:
            # Convert it into float
            val = float(var)
            return True
        except ValueError:
            return False

# Check variable for int
# Returns True if conversion was successful
# or False when the variable cannot be converted to an integer number
def check_int(var):
    try:
        val = int(var)
        return True
    except ValueError:
        return False
    
# Check variable for float
# Returns the variable as int if conversion was successful
# or False when the variable cannot be converted to an integer number
def check_float(var):
    try:
        # Convert it into integer
        val = int(var)
        return False
    except ValueError:
        try:
            # Convert it into float
            val = float(var)
            return True
        except ValueError:
            return False

##################
# Misc functions #
##################

# Function creates all working folders in the root directory of the program
# If they do not exist yet!
def create_prg_folders():
    # https://kodify.net/python/pathlib-path-mkdir-method/
    # Folder 'models'
    Path(setting["pth_models"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_yolo_models"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_custom_models"]).mkdir(parents=True, exist_ok=True)
    # Folder 'output'
    Path(setting["pth_output"]).mkdir(parents=True, exist_ok=True)
    # Folder 'predictions'
    Path(setting["pth_predictions"]).mkdir(parents=True, exist_ok=True)
    # Folder for training images and labels
    Path(setting["pth_training_images"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_validation_images"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_training_labels"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_validation_labels"]).mkdir(parents=True, exist_ok=True)

# Function to show if CUDA is working and software versions
def show_cuda_and_versions():
        print("\n>> DEVICE:")
        device = 'gpu (cuda)' if torch.cuda.is_available() else 'cpu (no cuda!)'
        print("Using Device:", device)
        print(">> VERSIONS:")
        print("Python: ", sys.version, "")
        print("Pytorch:", torch.__version__)
        print("CUDA:", torch.version.cuda)
        print("Ultralytics YOLO:", ultralytics.__version__)
        print("Supervision:", supervision.__version__)
        print("Opencv:", cv2.__version__)

# Function to exit any menue
def exit_menu(var, stop = "<exit>"):
    if(var == stop):
        print("Input canceled!")
        return True
    else:
        return False
    
# Prints a message to exit a menue with <exit>
def exit_menu_msg(stop = "<exit>"):
    print(f"> Enter {stop} to return to menue")

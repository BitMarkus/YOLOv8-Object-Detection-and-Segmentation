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

# Own modules
from train import Train 
from video import VideoOD
from image import ImageOD
from screen import ScreenOD
from batch_test import Batch_Pred
from img_split import Img_Split
import functions as fcn
from settings import setting

def main():

    # Show if CUDA is working and software versions
    fcn.show_cuda_and_versions()
    # Create program folders which are ignored by git
    fcn.create_prg_folders()    

    #############
    # Main Menu #
    #############

    while(True):  
        print("\n:MAIN MENU:")
        print(">> TRAIN:")
        print("  1) Train model")
        print(">> IMAGE PROCESSING:")
        print("  2) Object Detection")
        print("  3) Object Segmentation")
        print(">> VIDEO PROCESSING:")
        print("  4) Object Detection")
        print("  5) Object Segmentation")
        print("  6) Object Tracking")
        print(">> SCREEN CAPTURE PROCESSING:")
        print("  7) Object Detection")
        print("  8) Object Segmentation")
        print("  9) Object Tracking")
        print(">> TOOLS:")
        print("  10) Batch Testing Detection Models")
        print("  11) Dataset Splitter")
        print(">> PROGRAM:")
        print("  12) Exit Program")
        menu1 = int(fcn.input_int("Please choose: "))

        #################
        # Train Network # 
        #################

        if(menu1 == 1):       
            print("\n:TRAIN NETWORK:") 
            # Create train object
            od_train = Train()
            # Train model
            od_train()

        ###########################
        # Object Detection Images #  
        ###########################

        elif(menu1 == 2):      
            print("\n:OBJECT DETECTION ON IMAGES:") 
            od_img = ImageOD('detect')
            od_img()

        ##############################
        # Object Segmentation Images #  
        ##############################

        elif(menu1 == 3):      
            print("\n:OBJECT SEGMENTATION ON IMAGES:") 
            od_img = ImageOD('segment')
            od_img()

        ###########################
        # Object Detection Videos #  
        ###########################

        elif(menu1 == 4):      
            print("\n:OBJECT DETECTION ON VIDEOS:") 
            od_video = VideoOD('detect')
            od_video()

        ##############################
        # Object Segmentation Videos #  
        ##############################

        elif(menu1 == 5):      
            print("\n:OBJECT SEGMENTATION ON VIDEOS:") 
            od_video = VideoOD('segment')
            od_video()

        ##########################
        # Object Tracking Videos #  
        ##########################

        elif(menu1 == 6):      
            print("\n:OBJECT TRACKING ON VIDEOS:") 
            od_video = VideoOD('track')
            od_video()

        ###########################
        # Object Detection Screen #  
        ###########################

        elif(menu1 == 7):      
            print("\n:OBJECT DETECTION ON SCREEN:") 
            od_video = ScreenOD('detect')
            od_video()

        ##############################
        # Object Segmentation Screen #  
        ##############################

        elif(menu1 == 8):      
            print("\n:OBJECT SEGMENTATION ON SCREEN:") 
            od_video = ScreenOD('segment')
            od_video()

        ##########################
        # Object Tracking Screen #  
        ##########################

        elif(menu1 == 9):      
            print("\n:OBJECT TRACKING ON SCREEN:") 
            od_video = ScreenOD('track')
            od_video()

        #########
        # Tools #  
        #########

        elif(menu1 == 10):      
            print("\n:BATCH TESTING OF MODELS:") 
            batch_pred = Batch_Pred()
            batch_pred()

        elif(menu1 == 11):      
            print("\n:DATASET SPLITTER:") 
            img_split = Img_Split()
            img_split()

        ################
        # Exit Program #  
        ################      

        elif(menu1 == 12):
            print("\nExit program...")
            break
        
        # Wrong Input
        else:
            print("Not a valid option!")   

if __name__ == "__main__":
    main()
# Own modules
import functions as fcn
from od_img import ObjectDetectionImg
from od_vid import ObjectDetectionVid
from train import Train 
import functions as fnc
from settings import setting
from video import VideoOD
from image import ImageOD

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
        print("1) Train Network")
        print("2) Object detection on Images")
        print("3) Object detection on Videos")
        print("4) Object counting/tracking on Videos")
        print("5) Object segmentation on Videos")
        print("6) Exit Program")
        menu1 = int(fnc.input_int("Please choose: "))

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
            od_img = ImageOD()
            od_img()

        ###########################
        # Object Detection Videos #  
        ###########################

        elif(menu1 == 3):      
            print("\n:OBJECT DETECTION ON VIDEOS:") 
            od_video = VideoOD()
            od_video()

        ##################################
        # Object Cunting/Tracking Videos #  
        ##################################       

        elif(menu1 == 4):      
            print("\n:OBJECT COUNTING/TRACKING ON VIDEOS:") 
            # Create object datection object for videos
            od_detect = ObjectDetectionVid('count')
            # Print model classes
            od_detect.print_classes()
            # Execute detection
            od_detect()  

        ##############################
        # Object Segmentation Videos #  
        ############################## 

        elif(menu1 == 5):      
            print("\n:OBJECT SEGMENTATION ON VIDEOS:") 
            # Create object detection object for videos
            od_detect = ObjectDetectionVid('segm')
            # Print model classes
            od_detect.print_classes()
            # Execute detection
            od_detect()  

        ################
        # Exit Program #  
        ################      

        elif(menu1 == 6):
            print("\nExit program...")
            break
        
        # Wrong Input
        else:
            print("Not a valid option!")   

if __name__ == "__main__":
    main()
# Own modules
from train import Train 
from video import VideoOD
from image import ImageOD
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
        print(">> PROGRAM:")
        print("  7) Exit Program")
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

        ################
        # Exit Program #  
        ################      

        elif(menu1 == 7):
            print("\nExit program...")
            break
        
        # Wrong Input
        else:
            print("Not a valid option!")   

if __name__ == "__main__":
    main()
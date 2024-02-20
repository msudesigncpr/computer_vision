import cv2
import numpy as np
import os
import random
from scipy.stats import norm
from ultralytics import YOLO
import datetime


# dont worry about this one. it'll be what i use to make the pinhole function
def calculate_avg_x_y(img_file_path, margin = 20):
     # -----------------------------------------------LOAD IMAGE AND PROPERTIES------------------
    img = cv2.imread(img_file_path)
    # Check if the image was loaded successfully
    if img is None:
        print("Error: Could not read image file")
        exit()
    
    img_width = img.shape[1]
    img_height = img.shape[0]
    x = 0.5
    y = 0.5
    cropped_image_width = img_width * margin
    cropped_image_height = img_height * margin

    # -----------------------------------------------CROP & GRAY---------------------------------
    cropped_image = img[int(y-cropped_image_height) : int(y+cropped_image_height) , int(x-cropped_image_width) : int(x+cropped_image_width)]
    if cropped_image is None:
        print("Error: Could not crop image")
        exit()
    
    cv2.imshow("cropped image", cropped_image)
    cv2.waitKey(100)

    # check if there is channel dimension, grayscale if necesary 
    if len(cropped_image.shape) > 2:
        gray_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) 
    else:
        gray_cropped_image = cropped_image

    # -----------------------------------------------THRESHOLD BINERIZATION----------------------
    hist = cv2.calcHist([gray_cropped_image], [0], None, [256], [0, 256])
    hist = hist.ravel()
    z = np.linspace(0, 255, 256)
    param = norm.fit(z, loc=np.mean(hist), scale=np.std(hist))
    mean, std_dev = param
    k = .7 
    threshold = int(mean - k * std_dev)
    binary_image = cv2.threshold(gray_cropped_image, threshold, 255, cv2.THRESH_BINARY)[1]
    # binary_image = cv2.bitwise_not(binary_image)     #invert

    cv2.imshow("binary image", binary_image)
    cv2.waitKey(100)

    #-----------------------------------------------AVERAGE (x,y) PIXEL POSITIONS----------------------
    row_sums = np.sum(binary_image, axis=1)
    column_sums = np.sum(binary_image, axis=0)

    # Calculate row and column positions
    row_positions = np.arange(binary_image.shape[0])
    column_positions = np.arange(binary_image.shape[1])

    # Compute the total row and column sums
    total_row_sum = np.sum(row_sums)
    total_column_sum = np.sum(column_sums)

    # Calculate the average row and column positions
    average_row_position = np.dot(row_positions, row_sums) / (total_row_sum * cropped_image_width * 2)
    average_column_position = np.dot(column_positions, column_sums) / (total_column_sum * cropped_image_height * 2)

    print("Average Column Position: ", average_column_position, "Average Row Position: ", average_row_position)

    vertical_line_start_point = (int(average_column_position * cropped_image_width), 0)
    vertical_line_end_point = (int(average_column_position * cropped_image_width), cropped_image_height)

    horizontal_line_start_point = (0, int(average_row_position*cropped_image_height))
    horizontal_line_end_point = (cropped_image_width, int(average_row_position * cropped_image_height))

    print("vertical line start:",vertical_line_start_point)
    print("vertical line end:",vertical_line_end_point)

    print("horizontal line start:", horizontal_line_start_point)
    print("horizontal line end:", horizontal_line_end_point)

    cv2.line(cropped_image, vertical_line_start_point, vertical_line_end_point, (255, 0, 0), 1)
    cv2.line(cropped_image, horizontal_line_start_point, horizontal_line_end_point, (255,0,0), 1)
    cv2.imshow("final image" , cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def parse_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = [line.strip().split() for line in lines]
    return data

############################################################################################################ -- MOVE YOLO STUFF --
# moves all files in ./runs/detect to destination_folder_path

def move_YOLO_stuff(destination_folder_path):
    try:

        # date and time
        now = datetime.datetime.now()
        now = str(now)
        #remove spaces
        now = now.replace(" ", "_")
        #remove colons
        now = now.replace(":", "-")
        #remove periods
        now = now.replace(".", "-")

        destination_folder_path = os.path.join(destination_folder_path,now)
        print("Moving YOLO stuff to: " + destination_folder_path)

        # check if destination_folder_path exists, if not create it
        if not os.path.exists(destination_folder_path):
            os.makedirs(destination_folder_path)

        # move all files in ./runs/detect to destination_folder_path
        for file in os.listdir('./runs/detect'):
            os.rename(os.path.join('./runs/detect', file), os.path.join(destination_folder_path, file))


    except Exception as e:
        print("An error occured while moving YOLO stuff + " + str(e))


############################################################################################################ -- RESIZE IMAGES --
# Resizes all images in the specified folder to 640x640 and saves them in a new folder called 'resized' in the same directory.

def resize_images(image_folder_path):
    try:
        # resize all images in image_folder_path
        for image in os.listdir(image_folder_path):
            img = cv2.imread(os.path.join(image_folder_path, image), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (640, 640))
            #check if processed folder exists, if not create it
            if not os.path.exists(os.path.join(image_folder_path, 'resized')):
                os.makedirs(os.path.join(image_folder_path, 'resized'))
            cv2.imwrite(os.path.join(image_folder_path, 'resized', image), img)
    except Exception as e:
        print("An error occured while resizing images + " + str(e))


############################################################################################################ -- PROCESS PETRI DISH IMAGE --
# Processes all images in the specified folder using the YOLO model and binary discrimination. 
# The results are saved in a new folder called 'output' in the same directory.
        
# Parameters:
# - image_folder_path: Path to the folder containing the images.
# - good_colony_coord_output_path: Path to the folder where the good colony coordinates are written.
# - minimum_colony_distance: Minimum distance between two colonies.
# - model_sensitivity: Sensitivity of the YOLO model. 
# !!!! the lower the number, the more colonies are detected !!!!

def process_petri_dish_image(image_folder_path, good_colony_coord_output_path,  minimum_colony_distance = 0.03, model_sensitivity = 0.1, raw_yolo_dump_path = './yolo_dump'):
    try:
        #resize all the images to 640x640
        resize_images(image_folder_path)
        resized_image_folder_path = os.path.join(image_folder_path, 'resized')

        # run model on all images in image_folder_path
        model = YOLO('./models/norb_v3.11.2.pt')
        for image in os.listdir(resized_image_folder_path):
            model.predict(os.path.join(resized_image_folder_path,image), conf = model_sensitivity, save=True, imgsz=640, save_txt = True, classes = None, save_conf = True)  

        # this is where yolo puts the predictions for all the images in image_folder_path
        label_folder_path = './runs/detect/predict/labels'

        for label in os.listdir(label_folder_path):
            # label file name
            base_file_name = os.path.splitext(os.path.basename(label))[0]

            # call that big ass discimination function.
            discriminate(image_file_path=os.path.join(image_folder_path,base_file_name +'.jpg'), prediction_file_path= os.path.join(label_folder_path, label), good_output_path=good_colony_coord_output_path, min_distance=minimum_colony_distance)
        
        # move the stuff in label_folder_path to raw_yolo_dump_path
        move_YOLO_stuff(raw_yolo_dump_path)

        #remove all the images in resized folder, and the resized folder
        for file in os.listdir(resized_image_folder_path):
            os.remove(os.path.join(resized_image_folder_path, file))

        os.rmdir(resized_image_folder_path)

    except Exception as e:
        print("An error occured while processing petri dish image + " + str(e))

############################################################################################################ -- CREATE METADATA --
#creates:
# 1) images of each of the colonies in the txt file 
# 2) images of the petri dish with circles around the colonies

# Parameters:
# image_folder_path: Path to the folder containing the images
# colony_coords_folder_path: Path to the folder containing the colony coordinates
# metadata_output_path: Path to the folder where the metadata is written
# create_petri_dish_view: Boolean that determines whether or not to create a petri dish view
# create_colony_view: Boolean that determines whether or not to create a colony view

# File structure:
# - metadata_output_path
#   - colony_view
#     - [image name]
#       - sample_[colony number].jpg
#       - ...
#    - petri_dish_view
#      - [image name].jpg
#      - ...
#    note: currently if you call this function twice, it will not overwrite the previous metadata 
#    however, once sarah is appending the well number to the end of each line in the txt file,
#    this will be fixed. this is because the colony number is currently just a random number


def create_metadata(image_folder_path, colony_coords_folder_path, metadata_output_path = './metadata', create_petri_dish_view = False, create_colony_view = False):
    try:
        
        if not create_colony_view and not create_petri_dish_view:
            print("No metadata created. Please specify whether to create a petri dish view, a colony view, or both. Dumb idiot.")
            return

        # check if output path exists, if not create it
        if not os.path.exists(metadata_output_path):
            os.makedirs(metadata_output_path)
        
        colony_view_output_path = os.path.join(metadata_output_path, 'colony_view')
        petri_dish_view_output_path = os.path.join(metadata_output_path, 'petri_dish_view')

        if not os.path.exists(colony_view_output_path) and create_colony_view:
            os.makedirs(colony_view_output_path)
            print("Creating colony view metadata and putting it in: " + colony_view_output_path)

        if not os.path.exists(petri_dish_view_output_path) and create_petri_dish_view:
            os.makedirs(petri_dish_view_output_path) 
            print("Creating petri dish view metadata and putting it in: " + petri_dish_view_output_path)


  
        # Loop through each image file in the specified folder path
        for image_file in os.listdir(image_folder_path):
            
            # Print the path of the current image file
            
            # Read the image using OpenCV
            image = cv2.imread(os.path.join(image_folder_path, image_file))

            # Check if the image was successfully loaded
            # Extract the base file name of the image without the extension
            base_file_name = os.path.splitext(os.path.basename(image_file))[0]
            
            # Get the width and height of the image
            image_width = image.shape[1]
            image_height = image.shape[0]

            # Check if the output metadata folder exists, if not create it
            if not os.path.exists(metadata_output_path):
                os.makedirs(metadata_output_path)

            # Open the colony coordinates text file corresponding to the current image
            with open(os.path.join(colony_coords_folder_path, base_file_name + '.txt')) as file:
                # Read each line of the text file
                colony_lines = file.readlines()

                # bing is the name of the image. colony view images are placed in the path:
                # metadata_output_path/colony_view/[image name]/sample_[colony number].jpg
                bing = (os.path.join(colony_view_output_path, base_file_name))
                if not os.path.exists(bing):
                    os.makedirs(bing)

                if create_colony_view:
                    
                    bing = os.path.join(colony_view_output_path,base_file_name)
                    print("Making directory: ", bing)
                    if not os.path.exists(bing):
                        os.makedirs(bing)



                # Iterate over each line in the text file
                for colony_line in colony_lines:
                    # Split the line into individual elements based on whitespace
                    elements = colony_line.split()
                    # Extract the x, y, h, and w coordinates from the line and convert them to integer values
                    x = int(float(elements[1]) * image_width)
                    y = int(float(elements[2]) * image_height)
                    h = int(float(elements[3]) * image_height) 
                    w = int(float(elements[4]) * image_width)
                    # colony_number = int(elements[6])                                                          # SARAH: append the colony number (well letter/number) to the end of every line. 
                                                                                                                # this will get used below, but its just a random number for now
                    r = int(h/2)

                    if create_colony_view or create_petri_dish_view:
                        cv2.circle(image, (x, y), r, (0, 255, 0), 1)

                    # If specified, create a petri dish view
                    if create_petri_dish_view:
                        # Draw a small quare at the center of the colony
                        cv2.rectangle(image, (int(x-2), int(y-2)), (int(x+2), int(y+2)), (0, 0, 255), 1)

                        #TODO add a box that indicates where the needle could have gone

                    
                    # If specified, create a colony view
                    if create_colony_view:
                        # Crop the image to focus on the colony area
                        cropped_image = image[int(y-h) : int(y+h) , int(x-w) : int(x+w)]
                        # random number
                        colony_number = random.randint(1, 1000)                                                     #SARAH: this will be replaced with the colony number above, once you start sending me back files with that

                        cv2.imwrite(os.path.join(bing, '_' + str(colony_number) + '.jpg'), cropped_image)


                if create_petri_dish_view:
                    # Write the modified image with the colony annotations to the output metadata folder
                    cv2.imwrite(os.path.join(petri_dish_view_output_path, base_file_name + '.jpg'), image)
    
    except Exception as e:
        print("An error occured while creating metadata + " + str(e))

          


############################################################################################################ --ADD HOUGH CIRCLES--
#creates a .txt file with coordinates / size of colonies detected by hough circles
def add_hough_circles(image, prediction_file_path, display=False, display_time=5000):
    if image is None:
        print("Error: Could not load the image")
        exit()

    # make image grayscale if needed 
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (3, 3), 0)                                                                               ##PARAM                  

    # Apply Canny edge detector
    edges = cv2.Canny(blurred, 50, 60)                                                                                        ##PARAM

    # Use Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=15, minRadius=5, maxRadius=30)   ##PARAM
    
    #clear contents of output.txt
    open(prediction_file_path, 'w').close()

    #plot circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if display: 
                cv2.circle(image, (x, y), r, (255, 0, 0), 1)
            image_width = image.shape[1]
            image_height = image.shape[0]    
            with open (prediction_file_path, 'a') as f:
                f.write("0 " + str(x/image_width) + " " + str(y/image_height) + " " + str(r/image_width) +  " " + str(r/image_width) + " .07" "\n")
    else:
        print("No hough circles detected")

    if display:
        image = cv2.resize(image, (640, 640))
        cv2.imshow('Result', image)
        cv2.waitKey(display_time)
        cv2.destroyAllWindows()


#function to determine the distance from the center of the image to the center of the box
def distance(x0, y0, r0=0, x1=.5, y1=.5, r1=0):
    x_dist = abs(x0 - x1) ** 2
    y_dist = abs(y0 - y1) ** 2
    distance = (x_dist + y_dist) ** .5
    distance = distance - (r0 + r1)
    return (x_dist + y_dist) ** .5
        
############################################################################################################ --ADD HOUGH CIRCLES--
#creates a .txt file with coordinates / size of colonies detected by hough circles
#same format as yolo .txt files, so height and width are both just the radius
#puts it in prediction_output_path

# Parameters:
# - image_path: Path to the image file.
# - prediction_path: Path to the file where the predictions are written.
# - margin: Multiplier for the radius of the circle. This is useful for binerization stuff, because
#   yolo boxes tend to be bigger than the actual colony and hough circles tend to be smaller.
# - output_confidence: The confidence written to the .txt file. This is useful for showPrediction stuff.
# - display: Boolean that determines whether or not to display the image with the hough circles.
# - display_time: Time in milliseconds that the image is displayed for.
# - hough_confidence: The confidence appended to the end of each line in the .txt file 

# Creates:
# - Text file containing coordinates to all detected colonies.

def add_hough_circles(image_path, 
                      prediction_path, 
                      margin = 1, 
                      output_confidence = ".9", 
                      display=False, 
                      display_time=5000,
                      hough_confidence = 15
                      ):
    img = cv2.imread(os.path.join(image_path) , cv2.IMREAD_COLOR)
    if img is None:
        print("Error: Could not load the image")
        exit()

    # make image grayscale if needed 
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # detect circles
    blurred = cv2.GaussianBlur(img, (2, 2), 0)          #PARAM
    edges = cv2.Canny(blurred, 50, 60)                                                                                        ##PARAM
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=hough_confidence, minRadius=5, maxRadius=30)   ##PARAM
    
    #clear contents of output.txt
    open(prediction_path, 'w').close()

    #plot circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

    if display:
        img = cv2.imread(os.path.join(image_path) , cv2.IMREAD_COLOR)

    for (x, y, r) in circles:
        image_width = img.shape[1]
        image_height = img.shape[0]    
        with open (prediction_path, 'a') as f:
            f.write("0 " + str(x/image_width) + " " + str(y/image_height) + " " + str(r * margin/image_width) +  " " + str(r* margin/image_width) + " " + output_confidence + "\n")
        
        if display:               
            cv2.circle(img, (x, y), (r * margin), (255, 0, 0), 1)
    if display:
        img = cv2.resize(img, (640, 640))
        cv2.imshow('Result', img)
        cv2.waitKey(display_time)
        cv2.destroyAllWindows()

    else:
        print("No hough circles detected")

############################################################################################################ --BINARY DISCRIMINATE--
# Parameters:
# - img_file_path: Path to the image file.
# - x: x coordinate of the center of the colony.
# - y: y coordinate of the center of the colony.
# - width: Width of the colony.
# - height: Height of the colony.
# - margin: Multiplier for the width and height of the colony. This is useful for binerization stuff, because
#   yolo boxes tend to be bigger than the actual colony and hough circles tend to be smaller.
# - erosion_thresholds: Tuple of thresholds for each erosion iteration. If the normalized intensity of the binerized image
#   is above the threshold at any iteration, the colony is bad.
# - erosion_iterations: Tuple of iterations for each erosion. The number of iterations determines how much the colony is eroded.
# - original_display: Boolean that determines whether or not to display the original image.
# - bad_display: Boolean that determines whether or not to display the image if the colony is bad.
# - good_display: Boolean that determines whether or not to display the image if the colony is good.
# - display_time: Time in milliseconds that the image is displayed for.
# - save_folder_path: Path to the folder where the good colonies are saved.

# Returns:
# - Boolean that determines whether or not the colony is good.

def binary_disciminate(img_file_path, x, y, width, height, margin = .5, erosion_thresholds = (25, 999, 999, 999), erosion_iterations = (0, 1, 2, 3), original_display = False, bad_display = False, good_display=False, display_time = 2000, save_folder_path = None):
    try:

        # -----------------------------------------------LOAD IMAGE AND PROPERTIES------------------
        img = cv2.imread(img_file_path)
        # Check if the image was loaded successfully
        if img is None:
            print("Error: Could not read image file")
            exit()
        
        img_width = img.shape[1]
        img_height = img.shape[0]
        x = img_width * x
        y = img_height * y
        width = img_width * width * margin
        height = img_height * height * margin

        # -----------------------------------------------CROP & GRAY---------------------------------
        cropped_image = img[int(y-height) : int(y+height) , int(x-width) : int(x+width)]
        if cropped_image is None:
            print("Error: Could not crop image")
            exit()

        if len(cropped_image.shape) > 2:
            gray_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_cropped_image = cropped_image

        # -----------------------------------------------THRESHOLD BINERIZATION----------------------
        hist = cv2.calcHist([gray_cropped_image], [0], None, [256], [0, 256])
        hist = hist.ravel()
        z = np.linspace(0, 255, 256)
        param = norm.fit(z, loc=np.mean(hist), scale=np.std(hist))
        mean, std_dev = param
        k = .5 
        threshold = int(mean - k * std_dev)
        binary_image = cv2.threshold(gray_cropped_image, threshold, 255, cv2.THRESH_BINARY)[1]
        binary_image = cv2.bitwise_not(binary_image)                                                #invert                                 
        title = ""
        
        if original_display:
            title = "Original"
            display_image = cv2.resize(cropped_image, (640, 640))
            cv2.imshow(title, display_image)
            cv2.waitKey(display_time)
        
        
        #-----------------------------------------------AVERAGE (x,y) PIXEL POSITIONS----------------------
        row_sums = np.sum(binary_image, axis=1)
        column_sums = np.sum(binary_image, axis=0)

        # Calculate row and column positions
        row_positions = np.arange(binary_image.shape[0])
        column_positions = np.arange(binary_image.shape[1])

        # Compute the total row and column sums
        total_row_sum = np.sum(row_sums)
        total_column_sum = np.sum(column_sums)

        # Calculate the average row and column positions
        average_row_position = np.dot(row_positions, row_sums) / (total_row_sum * width * 2)
        average_column_position = np.dot(column_positions, column_sums) / (total_column_sum * height * 2)

        MAX_ROW_OFFSET = .4
        MAX_COLUMN_OFFSET = .4

        row_offset = abs(MAX_ROW_OFFSET - average_row_position)
        column_offset = abs(MAX_COLUMN_OFFSET - average_column_position)

        


        # -----------------------------------------------EROSION----------------------

        random_lil_fucker = "R-" + str(random.randint(1, 1000)) + "-"
        for l in range(len(erosion_iterations)):
            iterations = erosion_iterations[l]
            erosion_threshold = erosion_thresholds[l]
            eroded_binary_image = cv2.erode(binary_image, np.ones((2,2), np.uint8), iterations=iterations)
            binary_image_sum = np.sum(eroded_binary_image)          #sum of all pixels in the image
            binary_image_height = binary_image.shape[0]
            binary_image_width = binary_image.shape[1]
            binary_image_sum = binary_image_sum / (binary_image_width * binary_image_height)  #normalize
            if original_display:            
                title = "Erosion iteration: " + str(iterations) + " Normalized intensity: " + str(int(binary_image_sum)) + " Threshold: " + str(erosion_threshold)
                display_image = cv2.resize(eroded_binary_image, (640, 640))
                cv2.imshow(title, display_image)
                cv2.waitKey(display_time)
                cv2.destroyAllWindows()

            # if first iteration, use average row and col offset to return false if psat
                
                            # save binarized and original colony to binary save path
            if save_folder_path is not None:
                if not os.path.exists(os.path.join(save_folder_path, "good_colonies")):
                    os.makedirs(os.path.join(save_folder_path, "good_colonies"))


            if save_folder_path is not None:
                if not os.path.exists(os.path.join(save_folder_path, "bad_colonies")):
                    os.makedirs(os.path.join(save_folder_path, "bad_colonies"))


            if (binary_image_sum > erosion_threshold) or (row_offset > MAX_ROW_OFFSET) or (column_offset > MAX_COLUMN_OFFSET):
                # print("Colony exceed threshold at erosion iteration: " + str(iterations) + " Normalized intensity: " + str(int(binary_image_sum)) + " Threshold: " + str(erosion_threshold))

                if(bad_display):
                    title = random_lil_fucker + "BAD_ITR_" + str(iterations) + "_INT_" + str(int(binary_image_sum)) + "_TSH_" + str(erosion_threshold) + "_col_" + str(average_column_position) + "_row_" + str(average_row_position) #random shit is for testing
                    cv2.circle(img, (int(x), int(y)), int(width/margin), (0, 0, 255), 1)
                    img = cv2.resize(img, (640, 640))
                    # cv2.imshow(title, img)                      #show image of petri dish
                    # cv2.waitKey(display_time)
                    # cv2.destroyAllWindows()
                    display_image = cv2.resize(eroded_binary_image, (640, 640))
                    # cv2.imshow(title, display_image)            #show eroded image 
                    # cv2.waitKey(display_time)
                    # cv2.destroyAllWindows()



                    if save_folder_path is not None:
                        save_path = os.path.join(save_folder_path, "bad_colonies")
                        eroded_binary_image = cv2.resize(eroded_binary_image, (640, 640))
                        file_name = os.path.splitext(os.path.basename(img_file_path))[0]
                        binary_save_path = os.path.join(save_path, file_name + "_" + title + ".jpg")
                        cropped_save_path = os.path.join(save_path, random_lil_fucker + "cropped" + file_name + ".jpg")

                        # print("saving cropped good colony to: " + cropped_save_path)
                        # print("svaing binary good colony to: " + binary_save_path)
                        cv2.imwrite(binary_save_path, eroded_binary_image)
                        cv2.imwrite(cropped_save_path, cropped_image)
                return False
            
            else:

                #draw circles around colonies and show (binary_good_display in discriminate())
                #show binary image 
                if(good_display):
                    title = random_lil_fucker + "GOOD_ITR_" + str(iterations) + "_INT_" + str(int(binary_image_sum)) + "_TSH_" + str(erosion_threshold) + "_col_" + str(average_column_position) + "_row_" + str(average_row_position) #random shit is for testing
                    cv2.circle(img, (int(x), int(y)), int(width/margin), (0, 0, 255), 1)
                    img = cv2.resize(img, (640, 640))
                    # cv2.imshow(title, img)                      #show image of petri dish
                    # cv2.waitKey(display_time)
                    # cv2.destroyAllWindows()
                    display_image = cv2.resize(eroded_binary_image, (640, 640))
                    # cv2.imshow(title, display_image)            #show eroded image 
                    # cv2.waitKey(display_time)
                    # cv2.destroyAllWindows()


                    if save_folder_path is not None:
                        save_path = os.path.join(save_folder_path, "good_colonies")
                        eroded_binary_image = cv2.resize(eroded_binary_image, (640, 640))
                        file_name = os.path.splitext(os.path.basename(img_file_path))[0]
                        binary_save_path = os.path.join(save_path, file_name + "_" + title + ".jpg")
                        cropped_save_path = os.path.join(save_path, random_lil_fucker + "cropped" + file_name + ".jpg")

                        # print("saving cropped good colony to: " + cropped_save_path)
                        # print("svaing binary good colony to: " + binary_save_path)
                        cv2.imwrite(binary_save_path, eroded_binary_image)
                        cv2.imwrite(cropped_save_path, cropped_image)
            return True
    except Exception as e:
        print("An error occured while creating binary discriminating + " + str(e))
        

############################################################################################################ DISCRIMINATE
# takes in a prediction file and an image file
# creates two new files: good_colonies.txt and bad_colonies.txt
# good_colonies.txt contains the predictions that are good
# bad_colonies.txt contains the predictions that are bad
# there are myriad selection parameters that can be used to determine whether a colony is good or not
# the most important is the binary_discriminate function and the min distance parameter

# Parameters:
# - prediction_file_path: Path to the file containing predictions for all colonies.
# - image_file_path: Path to the image file.
# - good_output_path: Path to the file where the predictions for good colonies are written.
# - bad_output_path: Path to the file where the predictions for bad colonies are written.
# - min_distance: Minimum distance between two colonies.
# - min_selection_confidence: Minimum confidence for a colony to be selected.
# - min_discrimination_confidence: Minimum confidence for a colony to be used for discrimination.
# - min_size: Minimum size of a colony.
# - max_size: Maximum size of a colony.
# - maximum_ratio: Maximum ratio between width and height of a colony.
# - petri_dish_radius: Radius of the petri dish.

# Creates:
# - good_colonies.txt: File containing the predictions for good colonies.
# - bad_colonies.txt: File containing the predictions for bad colonies.

def discriminate(prediction_file_path, 
                 image_file_path,
                 BLACK = 'MAGIC',      #do not remove--code will break
                 good_output_path = None,
                 bad_output_path = None,
                 min_distance = .03,
                 min_selection_confidence = 0.14, 
                 min_discrimination_confidence = .05, 
                 min_size = .01, 
                 max_size = .5, 
                 maximum_ratio = .15, 
                 petri_dish_radius = .4,
                 binary_discrimination_margin = 2,
                 binary_bad_display = True,
                 binary_good_display = True,
                 binary_original_display = False,
                 display_time = 1000,
                 binary_save_folder_path = None
                 ):
    try:
        if good_output_path is None and bad_output_path is None:
            print("Error: Please specify the output paths for the good or bad colony location files")
            exit()

        base_file_name = os.path.splitext(os.path.basename(prediction_file_path))[0]

        if good_output_path is not None:
            if not os.path.exists(good_output_path):
                os.makedirs(good_output_path)
            good_file_name = os.path.join(str(good_output_path), base_file_name + '.txt')
            

            with open(good_file_name, 'w') as good_file:
                pass

        if bad_output_path is not None: 
            if not os.path.exists(bad_output_path):
                os.makedirs(bad_output_path)
            bad_file_name  = os.path.join(str(bad_output_path),  base_file_name + '.txt')

            with open(bad_file_name, 'w') as bad_file:
                pass

        # print("Base file name: " , base_file_name)
        # print("Good file name: " , good_file_name)
        # print("Bad file name: "  , bad_file_name)

        good_colonies = []
        bad_colonies = []

        #clear files


        with open(prediction_file_path) as predictionFile:
            lines = predictionFile.readlines()
            for main_colony_line in (lines):
                main_colony   = main_colony_line.split() # [class, x, y, width, height, confidence]
                main_colony_x = float(main_colony[1])
                main_colony_y = float(main_colony[2])
                main_colony_w = float(main_colony[3])
                main_colony_h = float(main_colony[4])
                main_colony_confidence = float(main_colony[5])
                ratio = abs((float(main_colony[4]) / float(main_colony[3])) - 1 ) #ok really this is how not square it is not the ratio but close enough
                 
                is_bad_colony = True
                if(distance(x0=main_colony_x, y0=main_colony_y) < petri_dish_radius and             # discriminate against colonies that are outside / near edge or petri dish 
                   main_colony_confidence > min_selection_confidence and                            # discriminate against colonies that are not confident enough       
                   binary_disciminate(img_file_path=image_file_path, x=main_colony_x, y=main_colony_y, width=main_colony_w,
                                        height=main_colony_h, original_display=binary_original_display, good_display = binary_good_display, bad_display=binary_bad_display, display_time=display_time, margin=binary_discrimination_margin, save_folder_path=binary_save_folder_path)):                    # discriminate against colonies that have too much shit near them
                    is_bad_colony = False
                    #iterate through all of the other colonies and check if there are any that are too close to the colony in question
                    for neighbor_colony_line in lines:                          
                        neighbor_colony = neighbor_colony_line.split()
                        neighbor_colony_x = float(neighbor_colony[1])
                        neighbor_colony_y = float(neighbor_colony[2])
                        neighbor_colony_r = float(neighbor_colony[3])
                        neighbor_colony_confidence = float(neighbor_colony[5])

                        distance_between_colonies = distance(x0=main_colony_x, y0=main_colony_y, r0=main_colony_w, 
                                                            x1=neighbor_colony_x, y1=neighbor_colony_y, r1=neighbor_colony_r)

                        if (distance_between_colonies <  min_distance and                #distance to colony
                            distance_between_colonies != 0.0 and                         #make sure it's not the same colony
                            neighbor_colony_confidence > min_discrimination_confidence): #make sure the colony prediction is confident enough to be used for discrimination
                            is_bad_colony = True

                #write bad colonies to bad_colonies.txt and good colonies to good_colonies.txt
                #only write the colony if it is not already in the file
                if is_bad_colony and bad_output_path is not None:
                    if not bad_colonies.__contains__(main_colony_line):
                        with open(bad_file_name, 'a') as bad_file:
                            bad_file.write(main_colony_line)
                            bad_colonies.append(main_colony_line)

                elif not is_bad_colony and good_output_path is not None:
                    if not good_colonies.__contains__(main_colony_line):
                        with open(good_file_name, 'a') as good_file:
                            good_file.write(main_colony_line)
                            lines.remove(main_colony_line)              # if the colony is good (for the most part this just means isolated), 
                                                                        # we do not have to worry about it so we can remove it from the list of lines 
                                                                        # being used to check if colonies are too close together

    except Exception as e:
        print("An error occured while discriminating + " + str(e))
        pass
                        
############################################################################################################ SHOW PREDICTIONS 
# Display colonies on an image based on prediction files

# Parameters:
# - good_colony_file_path: Path to the file containing predictions for good colonies. These appear as green circles
# - bad_colony_file_path: Path to the file containing predictions for bad colonies. These appear as red circles
# - image_path: Path to the image file.
# - display_time: Time in milliseconds the image is displayed.

def showPredictions(good_colony_file_path=None, bad_colony_file_path=None, image_path=None, display_image=False, display_time = 2000, save_folder_path = None):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load the image")
        exit()
    good_colony_counter = 0
    bad_colony_counter = 0 
    # GSD height: 0.179
    # GSD width: 0.142
    box_width = int(.4/.142)
    box_height = int(.4/.179)

    if good_colony_file_path is not None:
        with open(good_colony_file_path) as good_colony_file:
            good_colonies = good_colony_file.readlines()
            for colony_line in good_colonies:
                if colony_line is not None:
                    elements = colony_line.split()
                    x = int(float(elements[1]) * image.shape[1])
                    y = int(float(elements[2]) * image.shape[0])
                    r = int(float(elements[3]) * image.shape[1] / 2)
                    cv2.circle(image, (x, y), r, (0, 255, 0), 1)
                    cv2.rectangle(image, (int(x - box_width/2), int(y - box_height/2)), (int(x + box_width/2), int(y + box_height/2)), (0, 0, 255), 1)
                    cv2.putText(image, str(good_colony_counter), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
                    good_colony_counter = good_colony_counter + 1 

    if bad_colony_file_path is not None:
        with open(bad_colony_file_path) as bad_colony_file:
            bad_colonies = bad_colony_file.readlines()
            for colony_line in bad_colonies:
                if colony_line is not None:
                    elements = colony_line.split()
                    x = int(float(elements[1]) * image.shape[1])
                    y = int(float(elements[2]) * image.shape[0])
                    r = int(float(elements[3]) * image.shape[1] / 2)
                    cv2.circle(image, (x, y), r, (0, 0, 255), 1)
                    cv2.rectangle(image, (int(x - box_width/2), int(y - box_height/2)), (int(x + box_width/2), int(y + box_height/2)), (0, 0, 255), 1)
                    bad_colony_counter = bad_colony_counter + 1 

    print("Good colonies: " + str(good_colony_counter))
    print("Bad colonies: " + str(bad_colony_counter))
    
    image = cv2.resize(image, (640, 640))
    if display_image:
        cv2.imshow('image', image)
        cv2.waitKey(display_time)
        cv2.destroyAllWindows()


    if save_folder_path is not None:
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_folder_path, file_name + '.jpg')
        print("saving to: " + save_path)
        cv2.imwrite(save_path, image)
        cv2.waitKey(10)


############################################################################################################ --SHOW COLONIES--
# Display colonies one by one, up close 
def showColonies(prediction_file_path, image_path, display_time = 500, margin = 1):
    try:
        # -----------------------------------------------LOAD IMAGE AND PROPERTIES------------------
        img = cv2.imread(image_path)
        # Check if the image was loaded successfully
        if img is None:
            print("Error: Could not read image file")
            exit()
        img_width = img.shape[1]
        img_height = img.shape[0]

        with open(prediction_file_path) as file:
            colony_lines = file.readlines()
            for colony_line in colony_lines:
                elements = colony_line.split()
                x = int(float(elements[1]) * img_width)
                y = int(float(elements[2]) * img_height)
                w = int(float(elements[3]) * img_width * margin)
                h = int(float(elements[4]) * img_height * margin)

                # -----------------------------------------------CROP & GRAY---------------------------------
                cropped_img = img[int(y-h) : int(y+h) , int(x-w) : int(x+w)]
                if cropped_img is None:
                    print("Error: Could not crop image")
                    exit()
                
                box_width = int(1/.106)
                box_height = int(1/.134)

                #draw box with box_width and box_height
                cv2.rectangle(img, (int(x - box_width/2), int(y - box_height/2)), (int(x + box_width/2), int(y + box_height/2)), (0, 0, 255), 1)
                cv2.circle(img, (x, y), 1, (0, 0, 255), 1)
                cv2.circle(img, (x, y), int(.03 * img_width / 2), (0, 255, 0), 1) 
                
                display_img = cv2.resize(cropped_img, (640, 640))  
                cv2.imshow('Colony', display_img)
                cv2.waitKey(display_time)
                cv2.destroyAllWindows()
    except Exception as e:
        print("An error occured while showing colonies + " + str(e))
        pass

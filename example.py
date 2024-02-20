import CPR_tools as cpr
import datetime

# # control code has just taken images of 6 petri dishes. for this example, these images are in the images folder
# # call process_petri_dish_image to process the images. This function will output the coordinates of the good colonies in the good_colony_coords folder
# # it will also output the raw yolo dump in the yolo_dump folder. this is what seth can use to train a new model if he wants.
# # when I run the yolo model, it creates a text file for every image that has the coordinates to the colonies it has found, and places it in 
# # ./runs/detect/predict/labels this is what my code grabs and uses to do discimination for crowded colonies, and is why we have to deal with text files
# # this is what is getting moved to yolo_dump
# cpr.process_petri_dish_image(image_folder_path='./images/', good_colony_coord_output_path='./good_colony_coords/', raw_yolo_dump_path='./yolo_dump/')

# # after running process_petri_dish_image, your code would presumably take the 6 text files in good_colony_coords, select 96 colonies, append the number of the well each was destined for
# # at the end of the line (more on that in the comments, search for "sarah"), and then call this function on them.
# cpr.create_metadata(image_folder_path='./images/', colony_coords_folder_path='./good_colony_coords/', create_petri_dish_view=True, create_colony_view= True)

cpr.calculate_avg_x_y('./pinhole_good_test.jpg')
cpr.calculate_avg_x_y('./pinhole_bad_test.jpg')
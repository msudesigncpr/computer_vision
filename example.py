import CPR_tools as cpr
import datetime
import os
import shutil


shutil.rmtree('.\\runs\detect\predict\\')

cpr.process_petri_dish_image(image_folder_path='./white/', good_colony_coord_output_path='./good_colony_coords/', raw_yolo_dump_path=None, binary_save_folder_path='./binary_images/')


cpr.create_metadata(image_folder_path='./white/', colony_coords_folder_path='.\good_colony_coords\\', create_petri_dish_view=True, create_colony_view= True)




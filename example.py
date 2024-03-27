import CPR_tools as cpr
import datetime
import os
import cv2
import shutil


shutil.rmtree('.\\runs\detect\predict\\')
shutil.rmtree('.\\metadata\colony_view\\')
shutil.rmtree('.\\metadata\petri_dish_view\\')

cpr.process_petri_dish_image(image_folder_path='.\\ibs_ecoli\\', good_colony_coord_output_path='./good_colony_coords/', raw_yolo_dump_path=None, binary_save_folder_path='./binary_images/', model_sensitivity=0.01)


cpr.create_metadata(image_folder_path='.\\ibs_ecoli\\', colony_coords_folder_path='.\good_colony_coords\\', create_petri_dish_view=True, create_colony_view= True)


# cpr.process_petri_dish_image(image_folder_path='./red/', good_colony_coord_output_path='./good_colony_coords/', raw_yolo_dump_path=None, binary_save_folder_path='./binary_images/', minimum_colony_distance=0.001)


# cpr.create_metadata(image_folder_path='.\\red\\', colony_coords_folder_path='.\good_colony_coords\\', create_petri_dish_view=True, create_colony_view= True)
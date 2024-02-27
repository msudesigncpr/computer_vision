# Computer Vision!

## Main Functions

### process_petri_dish_image()
Pass this the path to a folder with petri dish images in it (**baseplatePhotos**), and it will find the colonies. The locations to these colonies, 
in fractional coordinates, will be deposited the path you specify (**good_colony_coords**). These coordinates will be in .txt files, with the same name as the petri dish image they correspond to. There are a number of sensitivity settings you can adjust as well.

### create_metadata() 
Pass this a path to a folder with petri dish images in it (**baseplatePhotos)**, along with the path to a folder that has (fractional) coordinates to colonies you have sampled from (**sampleColonies**), and it will create metadata for the samples. An example of what this looks like is shown in **metadata**. The **colony_view** images will be named with the well plate they were deposited in. The **petri_dish_view** images will be named with the petri dish image they correspond to. 

### pinhole()
Pass this the path to an image taken of the lit hole on the baseplate, and it will determine whether or not the camera was centered. You are able to change how far off the x/y axes are off by before the function returns false. You can also specify a path for the processed image to be saved at (**pinhole_test_images/pinhole_test.jpg**). 

## Folders and Files
baseplatePhotos - photos of petri dishes
good_colony_coords - .txt files with fractional coordinates to colonies in **baseplatePhotos** that are viable to be sampled from. 
metadata - metadata created from **create_metadata** function.
models - yolo models used by **process_petri_dish_image()**.
pinhole_test_images - image of lit pinhole, resultant processed image indicated where centroid of lit pinhole is, along with a box that represents how far off the x/y axes can be before being too far off (this is changeable in **pinhole()** parameters). 
sampleColonies - colonies that have been sampled from. Control code MUST append the well plate number to the end of each colony coordinate, or **create_metadata()** will break (for now).

---

CPR_tools.py - contains all functions. 
example.py - example of how to use main functions

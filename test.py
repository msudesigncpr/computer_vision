import cv2
import numpy as np
import os 


def change_image_contrast(input_image_path, output_image_path):
    for image in os.listdir(input_image_path):
        img = cv2.imread(os.path.join(input_image_path, image))    

        blurred_img = cv2.bilateralFilter(img, 9, 75, 75)          #PARAM

        # converting to LAB color space
        lab = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Applying CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9,9))
        cl = clahe.apply(l_channel)

        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl,a,b))

        # Converting image from LAB Color model to BGR color spcae
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # enhanced_img = blurred_img ########################################################### REMOVE LATER

        print(os.path.join(output_image_path, image))
        cv2.imwrite(os.path.join(output_image_path, image), enhanced_img)

if __name__ == '__main__':
    change_image_contrast('./ibs_ecoli\\unaltered', './ibs_ecoli\\altered')
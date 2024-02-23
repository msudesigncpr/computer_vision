import cv2
import numpy as np
import matplotlib.pyplot as plt
import CPR_tools as cpr


def pinhole(img_file_path, row_deviation_threshold = 0.1, column_deviation_threshold = 0.1, center_point = (0.5, 0.5), x_margin=0.07, y_margin = 0.09):
    # Reading image
    img = cv2.imread(img_file_path)
    if img is None:
        print("Error: Could not read image file")
        exit()

    img_width = img.shape[1]
    img_height = img.shape[0]    

    cropped_image_width = img_width * x_margin
    cropped_image_height = img_height * y_margin

    cropped_image_x = center_point[0] * img_width
    cropped_image_y = center_point[1] * img_height

    # Cropping the image
    cropped_image = img[int(cropped_image_y - cropped_image_height):int(cropped_image_y + cropped_image_height), 
                        int(cropped_image_x - cropped_image_width):int(cropped_image_x + cropped_image_width)]
    
    # Resizing the cropped image to 640 x 480
    cropped_image = cv2.resize(cropped_image, (640, 480))


    # create binary image
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # # display images 
    # img = cv2.resize(img, (640, 480))
    # cv2.imshow('Original Image', img)
    # cv2.waitKey(500)
    
    # cv2.imshow('Cropped Image', cropped_image)
    # cv2.waitKey(500)

    # disp_binary_img = cv2.resize(binary_image, (640, 480))
    # cv2.imshow('Binary Image', disp_binary_img)
    # cv2.waitKey(0)

    # find centroid
    y,x = np.nonzero(binary_image)
    average_column_position = x.mean() / binary_image.shape[1]
    average_row_position = y.mean() / binary_image.shape[0]

    # # plot the binary image and the average position (centroid) 
    # plt.figure()
    # plt.imshow(binary_image, cmap='gray')
    # plt.plot(average_column_position * binary_image.shape[1], average_row_position * binary_image.shape[0], 'r.') # yes that is stupid 
    # plt.title('Binary Image')
    # plt.show()


    # Calculating deviations
    column_deviation = abs(0.5 - average_column_position)
    row_deviation = abs(0.5 - average_row_position)

    # Defining line start and end points
    vertical_line_start_point = (int(average_column_position * cropped_image.shape[1]), 0)
    vertical_line_end_point = (int(average_column_position * cropped_image.shape[1]), cropped_image.shape[0])

    horizontal_line_start_point = (0, int(average_row_position * cropped_image.shape[0]))
    horizontal_line_end_point = (cropped_image.shape[1], int(average_row_position * cropped_image.shape[0]))


    # # Printing line points
    # print("vertical line start:", vertical_line_start_point)
    # print("vertical line end:", vertical_line_end_point)
    # print("horizontal line start:", horizontal_line_start_point)
    # print("horizontal line end:", horizontal_line_end_point)



    # Checking if deviations exceed thresholds
    if column_deviation > column_deviation_threshold or row_deviation > row_deviation_threshold:
        # Printing deviation details
        print("One or more deviation exceeds threshold")
        print("Column Deviation:", column_deviation, "Row Deviation:", row_deviation)
        print("Column Deviation Threshold:", column_deviation_threshold, "Row Deviation Threshold:", row_deviation_threshold)

        # Drawing lines and text on image
        cv2.line(cropped_image, vertical_line_start_point, vertical_line_end_point, (0, 0, 255), 1)
        cv2.line(cropped_image, horizontal_line_start_point, horizontal_line_end_point, (0, 0, 255), 1)
        cv2.putText(cropped_image, ("("+str(average_column_position)[:8] + ","), (int(0.1*cropped_image.shape[0]),int(0.1*cropped_image.shape[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(cropped_image, (str(average_row_position)[:8] + ")"), (int(0.1*cropped_image.shape[0]),int(0.2*cropped_image.shape[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # draw deviation threshold box
        cv2.rectangle(cropped_image, (int(0.5 * cropped_image.shape[1] - column_deviation_threshold * cropped_image.shape[1]), int(0.5 * cropped_image.shape[0] - row_deviation_threshold * cropped_image.shape[0])), (int(0.5 * cropped_image.shape[1] + column_deviation_threshold * cropped_image.shape[1]), int(0.5 * cropped_image.shape[0] + row_deviation_threshold * cropped_image.shape[0])), (0, 0, 255), 1)


        # show image
        cv2.imshow("final image", cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return False
    else:
        # Printing deviation details
        print("Both deviations are within threshold")
        print("Column Deviation:", column_deviation, "Row Deviation:", row_deviation)
        print("Column Deviation Threshold:", column_deviation_threshold, "Row Deviation Threshold:", row_deviation_threshold)

        # Drawing lines and text on image
        cv2.line(cropped_image, vertical_line_start_point, vertical_line_end_point, (0, 255, 0), 1)
        cv2.line(cropped_image, horizontal_line_start_point, horizontal_line_end_point, (0, 255, 0), 1)
        cv2.putText(cropped_image, ("("+str(average_column_position)[:8] + ","), (int(0.1*cropped_image.shape[0]),int(0.1*cropped_image.shape[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(cropped_image, (str(average_row_position)[:8] + ")"), (int(0.1*cropped_image.shape[0]),int(0.2*cropped_image.shape[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # draw deviation threshold box
        cv2.rectangle(cropped_image, (int(0.5 * cropped_image.shape[1] - column_deviation_threshold * cropped_image.shape[1]), int(0.5 * cropped_image.shape[0] - row_deviation_threshold * cropped_image.shape[0])), (int(0.5 * cropped_image.shape[1] + column_deviation_threshold * cropped_image.shape[1]), int(0.5 * cropped_image.shape[0] + row_deviation_threshold * cropped_image.shape[0])), (0, 255, 0), 1)

        # show image
        cv2.imshow("final image", cropped_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return True


# pinhole('./pinhole_original_test.jpg')
pinhole('./pinhole_lights_on.jpg', row_deviation_threshold=.1, column_deviation_threshold=.1, center_point=(0.5, 0.48))
# pinhole('./pinhole_lights_off.bmp')

import numpy as np
import RANSAC
import cv2


def stitch(image1, image2, hom, homInv):
    # Calculating the size of stitched image

    # GETTING THE CORNERS OF IMAGE2 (X, Y) REPRESENTATION OF PIXEL OF IMAGES

    # print(image1.shape)
    # print(image2.shape)

    h2, w2, d2 = image2.shape
    h1, w1, d1 = image1.shape
    # print(h2)
    # print(w2)

    left_top = np.array([0, 0])
    left_bottom = np.array([0, h2])
    right_top = np.array([w2, 0])
    right_bottom = np.array([w2, h2])

    # GETTING THE PROJECTIONS OF CORNERS OF IMAGE2
    p_x = []
    p_y = []
    p_left_top = RANSAC.project(left_top, homInv)
    p_x.append(p_left_top[0])
    p_y.append(p_left_top[1])

    p_left_bottom = RANSAC.project(left_bottom, homInv)
    p_x.append(p_left_bottom[0])
    p_y.append(p_left_bottom[1])

    p_right_top = RANSAC.project(right_top, homInv)
    p_x.append(p_right_top[0])
    p_y.append(p_right_top[1])

    p_right_bottom = RANSAC.project(right_bottom, homInv)
    p_x.append(p_right_bottom[0])
    p_y.append(p_right_bottom[1])

    p_x.sort()
    p_y.sort()

    # print(p_left_top)
    # print(p_left_bottom)
    # print(p_right_top)
    # print(p_right_bottom)
    #
    # print(p_x)
    # print(p_y)

    min_x = int(p_x[0])
    max_x = int(p_x[3])
    if max_x < w1:
        max_x = w1

    min_y = int(p_y[0])
    max_y = int(p_y[3])
    if max_y < h1:
        max_y = h1

    if min_x < 0:
        x_offset = abs(min_x)
    else:
        x_offset = 0

    if min_y < 0:
        y_offset = abs(min_y)
    else:
        y_offset = 0

    stitched_image = np.zeros((y_offset + max_y, x_offset + max_x, 3), dtype=np.uint8)
    # print(stitched_image.shape)
    # print(stitched_image.shape)
    # cv2.imshow('stitched_images', stitched_image)
    # cv2.waitKey(0)

    # PUTTING THE IMAGE1 INTO STITCHED IMAGE AT RIGHT LOCATION

    stitched_image[y_offset:h1 + y_offset, x_offset:w1 + x_offset] = image1
    # cv2.imshow('test', stitched_image)
    # cv2.waitKey(0)

    # PROJECTING IMAGE1 ONTO IMAGE2 SPACE
    h3, w3, d3 = stitched_image.shape
    for y in range(-y_offset, h3):
        for x in range(-x_offset, w3):
            point = np.array([x, y])
            projection = RANSAC.project(point, hom)
            p_x = int(projection[0])
            p_y = int(projection[1])

            if 0 < p_x < w2 and 0 < p_y < h2:
                if y + y_offset < h3 and x + x_offset < w3:
                    stitched_image[y + y_offset, x + x_offset] = image2[p_y, p_x]

    # cv2.imshow('final', stitched_image)
    # cv2.waitKey(0)

    return stitched_image


def stitchAll(image_to_stitch, sift, bf, num_iterations, inlier_threshold, number_of_points):
    for image in range(len(image_to_stitch)):
        if image == 0:
            image1 = image_to_stitch[image]
            image2 = image_to_stitch[image + 1]
            image = image + 1

        else:
            image1 = image3
            image2 = image_to_stitch[image]

        # print(image1.shape)
        # print(image2.shape)

        key_points_image_1, des_image_1 = sift.detectAndCompute(image1, None)
        key_points_image_2, des_image_2 = sift.detectAndCompute(image2, None)

        matches = bf.match(des_image_1, des_image_2)

        best_matches, hom, inv_hom = RANSAC.Ransac(num_iterations, inlier_threshold, number_of_points,
                                                   key_points_image_1,
                                                   key_points_image_2,
                                                   matches)

        image3 = stitch(image1, image2, hom, inv_hom)
        # print(image3.shape)
        # cv2.imshow(str(image), image3)
        # cv2.waitKey(0)

    return image3

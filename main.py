import cv2
import HarrisCorner
import RANSAC
import Stitch

# Step - 1

# Reading and converting the boxes.png image into gray scale
boxes = cv2.imread("project_images/Boxes.png")
gray_boxes = cv2.cvtColor(boxes, cv2.COLOR_BGR2GRAY)

# Detecting key points using harris corner matrix
key_points_boxes = HarrisCorner.computeHarrisCornerKeyPoints(gray_boxes)

output_boxes = cv2.drawKeypoints(boxes, key_points_boxes, boxes)
cv2.imshow("Boxes", output_boxes)
cv2.waitKey(0)
# cv2.imwrite("output/1a.png", output_boxes)

# Step - 2

# Creating sift object
sift = cv2.xfeatures2d_SIFT.create()

# Reading and converting the Rainier.png image into gray scale
image_1 = cv2.imread("project_images/Rainier1.png")
rainier1 = image_1.copy()
gray_rainier1 = cv2.cvtColor(rainier1, cv2.COLOR_BGR2GRAY)

# Detecting key points using sift
key_points_rainier1, des_rainier1 = sift.detectAndCompute(gray_rainier1, None)

cv2.drawKeypoints(rainier1, key_points_rainier1, rainier1)
cv2.imshow("Rainier1", image_1)
cv2.waitKey(0)
# cv2.imwrite("output/1b.png", rainier1)

# Reading and converting the Rainier.png image into gray scale
image_2 = cv2.imread("project_images/Rainier2.png")
rainier2 = image_2.copy()
gray_rainier2 = cv2.cvtColor(rainier2, cv2.COLOR_BGR2GRAY)

# Detecting key points using sift
key_points_rainier2, des_rainier2 = sift.detectAndCompute(gray_rainier2, None)

output_rainier2 = cv2.drawKeypoints(rainier2, key_points_rainier2, rainier2)
cv2.imshow("Rainier2", output_rainier2)
cv2.waitKey(0)
# cv2.imwrite("output/1c.png", output_rainier2)

# Drawing matches between key-points of rainier1 and rainier2 images
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(des_rainier1, des_rainier2)
# matches = sorted(matches, key=lambda x: x.distance)
matches_output = cv2.drawMatches(rainier1, key_points_rainier1, rainier2, key_points_rainier2, matches[:20], None)
cv2.imshow("Matches", matches_output)
cv2.waitKey(0)
# cv2.imwrite("output/2.png", matches_output)

# Step - 3

# Setting the iterations count and inlier_threshold to considered as inlier and number of points
num_iterations = 100
inlier_threshold = 10
number_of_points = 4


# Getting best matches between images, homography matrix and inverse homography matrix
best_matches, hom, invHom = RANSAC.Ransac(num_iterations, inlier_threshold, number_of_points, key_points_rainier1,
                                          key_points_rainier2,
                                          matches)
best_matches_output = cv2.drawMatches(rainier1, key_points_rainier1, rainier2, key_points_rainier2, best_matches[:20],
                                      None)
cv2.imshow("BestMatches", best_matches_output)
cv2.waitKey(0)
# cv2.imwrite("output/3.png", best_matches_output)

# Step - 4
# Stitching the images together
stitched_image = Stitch.stitch(image_1, image_2, hom, invHom)
cv2.imshow("4", stitched_image)
cv2.waitKey(0)
# cv2.imwrite("output/4.png", stitched_image)

# Stitching all 6 images of Rainier
image_to_stitch = [cv2.imread('project_images/Rainier1.png'), cv2.imread('project_images/Rainier2.png'),
                   cv2.imread('project_images/Rainier3.png'), cv2.imread('project_images/Rainier4.png'),
                   cv2.imread('project_images/Rainier5.png'), cv2.imread('project_images/Rainier6.png')]
image_3 = Stitch.stitchAll(image_to_stitch, sift, bf, num_iterations, inlier_threshold, number_of_points)
cv2.imshow("AllStitched", image_3)
cv2.waitKey(0)
# cv2.imwrite('output/AllStitched.png', image_3)

# Stitching the images I taken
image_to_stitch = [cv2.imread('project_images/image1.jpg'), cv2.imread('project_images/image2.jpg'),
                   cv2.imread('project_images/image3.jpg')]
image_3 = Stitch.stitchAll(image_to_stitch, sift, bf, num_iterations, inlier_threshold, number_of_points)
cv2.imshow("AllMyImages", image_3)
cv2.waitKey(0)
# cv2.imwrite('output/AllMyImages.png', image_3)

# Stitching Hanging images
image_to_stitch = [cv2.imread('project_images/Hanging1.png'), cv2.imread('project_images/Hanging2.png')]
image_3 = Stitch.stitchAll(image_to_stitch, sift, bf, num_iterations, inlier_threshold, number_of_points)
cv2.imshow('AllHanging', image_3)
cv2.waitKey(0)
# cv2.imwrite('output/AllHanging.png', image_3)

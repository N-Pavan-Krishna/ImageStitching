import cv2
import numpy as np
import random


# Gives us homography matrix and inverse homography matrix
def project(p, homography):
    # converting point into homogeneous coordinates
    # print(p)
    p = np.array([[p[0]], [p[1]], [1]])
    temp = np.dot(homography, p)  # .astype(np.int)

    # converting homogeneous coordinates back to normal coordinates
    if temp[2] != 0:
        temp = temp / temp[2]

    return np.array([temp[0], temp[1]])


# Counts the inliers
def computeInlierCount(query_key_points, train_key_points, matched_points, homography, inlier_count):
    # getting all the key-points from the query and train images
    query = np.array([query_key_points[m.queryIdx].pt for m in matched_points])
    train = np.array([train_key_points[m.trainIdx].pt for m in matched_points])
    count = 0
    # for every key point in query image find the projection in train image and finding the difference between computed
    # key point and key point in actual train image
    for index in range(len(query)):
        # getting the expected key point of the train image
        expected_point = project(query[index], homography)
        actual_point = np.array([[train[index][0]], [train[index][1]]])
        # print(expected_point)
        # print(actual_point)

        # Finding the distance between actual key point and expected key point
        ssd = np.sqrt(((expected_point - actual_point) ** 2)).sum()
        # print(ssd)
        if ssd <= inlier_count:
            count += 1
    return count


# Find the inliers
def getInliers(query_key_points, train_key_points, matched_points, max_homography, inlier_count):
    # getting all the key-points from the query and train images
    inliers = []
    query = np.array([query_key_points[m.queryIdx].pt for m in matched_points])
    train = np.array([train_key_points[m.trainIdx].pt for m in matched_points])
    for index in range(len(query)):
        # getting the expected key point of the train image
        expected_point = project(query[index], max_homography)
        actual_point = np.array([[train[index][0]], [train[index][1]]])
        ssd = np.sqrt(((expected_point - actual_point) ** 2)).sum()
        # print(ssd)
        if ssd <= inlier_count:
            inliers.append(matched_points[index])

    return inliers


# Returns the inliers, homography matrix and inverse-homography matrix
def Ransac(iterations, inlier_threshold, points, query_key_points, train_key_points, matched_points):
    # we repeat the process of finding best homography until number of iterations
    max_inlier_count = 0
    for i in range(iterations):

        # getting "point's" random matches from the matches
        random_matches = random.sample(matched_points, points)

        # getting key points of query and train images based on chosen random matches
        query = np.array([query_key_points[m.queryIdx].pt for m in random_matches])
        train = np.array([train_key_points[m.trainIdx].pt for m in random_matches])

        # For chosen random points getting the homography
        homography, mask = cv2.findHomography(query, train, method=0)

        # Using the homography getting the inlier counts
        inlier_count = computeInlierCount(query_key_points, train_key_points, matched_points, homography,
                                          inlier_threshold)
        # print(inlier_count)

        # Storing maximum inlier count and best homography
        if inlier_count > max_inlier_count:
            max_inlier_count = inlier_count
            max_homography = homography

    # print("Maximum inliers found : " + str(max_inlier_count))
    # print("Best Homography :- ")
    # print(max_homography)

    # getting inliers with best homography
    inliers = getInliers(query_key_points, train_key_points, matched_points, max_homography, inlier_threshold)
    inliers = sorted(inliers, key=lambda x: x.distance)
    # getting keypoints that are inliers
    query = np.array([query_key_points[i.queryIdx].pt for i in inliers])
    train = np.array([train_key_points[i.trainIdx].pt for i in inliers])
    # print(len(query))
    # print(len(train))
    # print(len(inliers))

    # calculating homography using all key points that are inliers
    max_homography, mask = cv2.findHomography(query, train, method=0)
    inverse_homography = np.linalg.inv(max_homography)

    return inliers, max_homography, inverse_homography

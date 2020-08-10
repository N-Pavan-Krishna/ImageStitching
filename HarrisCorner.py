import numpy as np
import cv2


def get_x_gradient(image):
    gradient_x = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0)
    return gradient_x


def get_y_gradient(image):
    gradient_y = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1)
    return gradient_y


def smooth_gradient(i_xx, i_yy, i_xy):
    i_xx = cv2.GaussianBlur(i_xx, (3, 3), 0, 0)
    i_yy = cv2.GaussianBlur(i_yy, (3, 3), 0, 0)
    i_xy = cv2.GaussianBlur(i_xy, (3, 3), 0, 0)
    return i_xx, i_yy, i_xy


def smooth_gradient(i_xx, i_yy, i_xy):
    i_xx = cv2.GaussianBlur(i_xx, (3, 3), 0, 0)
    i_yy = cv2.GaussianBlur(i_yy, (3, 3), 0, 0)
    i_xy = cv2.GaussianBlur(i_xy, (3, 3), 0, 0)
    return i_xx, i_yy, i_xy


def compute_response(i_xx, i_yy, i_xy, window_size):
    step = int(np.floor(window_size / 2))
    response = np.zeros((i_xx.shape[0], i_xx.shape[1]))
    key_points = []
    h, w = i_xx.shape
    for y in range(h):
        for x in range(w):
            if y - step > -1 and y + step + 1 < i_xx.shape[0] and x - step > -1 and x + step + 1 < i_xx.shape[1]:
                xx = i_xx[y - step:y + step + 1, x - step:x + step + 1].sum()
                yy = i_yy[y - step:y + step + 1, x - step:x + step + 1].sum()
                xy = i_xy[y - step:y + step + 1, x - step:x + step + 1].sum()
                r = (xx * yy - xy * xy) - 0.4 * (xx + yy)
                if r > 10:
                    response[y, x] = r
    return response


def non_max_suppression(response, window_size):
    step = int(np.floor(window_size / 2))
    interest_point = []
    # response = np.pad(response, (step, step), 'constant', constant_values=0)
    col, row = response.shape
    for x in range(step, row - step):
        for y in range(step, col - step):
            r = response[y - step: y + step + 1, x - step:x + step + 1]
            maximum = r.max()
            if response[y, x] == maximum and maximum != 0:
                interest_point.append(cv2.KeyPoint(x, y, _size=1))
    return interest_point


def computeHarrisCornerKeyPoints(image):
    image_x = get_x_gradient(image)
    image_y = get_y_gradient(image)
    image_xx = image_x * image_x
    image_yy = image_y * image_y
    image_xy = image_x * image_y

    image_xx, image_yy, image_xy = smooth_gradient(image_xx, image_yy, image_xy)

    response = compute_response(image_xx, image_yy, image_xy, 3)
    key_points_boxes = non_max_suppression(response, 11)

    return key_points_boxes


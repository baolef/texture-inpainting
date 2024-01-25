# Created by Baole Fang at 1/24/24

import numpy as np
import cv2


def compute_mask(input_path, output_path):
    # load image
    img = cv2.imread(input_path)

    # upper and lower bounds for gray
    lower_gray = np.array([110, 110, 110])
    upper_gray = np.array([160, 160, 160])

    # create mask for gray
    mask = cv2.inRange(img, lower_gray, upper_gray)

    # remove small connected components
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, None, None, None, 8, cv2.CV_32S)
    areas = stats[1:, cv2.CC_STAT_AREA]
    result = np.zeros(labels.shape, np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= 20:  # keep
            result[labels == i + 1] = 255

    # fill holes
    mask = np.zeros((result.shape[0] + 2, result.shape[1] + 2), dtype=np.uint8)
    holes = cv2.floodFill(result.copy(), mask, (0, 0), 255)[1]
    holes = ~holes
    result[holes == 255] = 255

    # save mask
    cv2.imwrite(output_path, result)


if __name__ == '__main__':
    compute_mask('aggregated_rgb.png', 'mask.png')

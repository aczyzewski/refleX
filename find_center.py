import numpy as np
import cv2 as cv
import pandas as pd
import sys

from math import cos, sin, log
from sklearn.metrics.pairwise import cosine_distances

import logging
import glob

import util


def get_radii_coordinates(img, center, num_samples=20, offset=0):

    """
    Example:
    ------
    >> sampling_vectors(img, (3, 3), 4, 0)
    [[(3, 3), (4, 3), (5, 3), (6, 3)],
     [(3, 3), (3, 4), (3, 5), (3, 6)],
     [(3, 3), (2, 3), (1, 3), (0, 3)],
     [(3, 3), (3, 2), (3, 1), (3, 0)]]

    """
    x, y = center
    if not 0 < x < img.shape[1] or not 0 < y < img.shape[0]:
        return []

    angles = np.linspace(0 + offset, 360 + offset, num_samples + 1)[:-1]
    radius = np.min([img.shape[1] - x - 1, x, img.shape[0] - y - 1, y])

    vectors = []
    for angle in angles:
        vector = []
        distances = np.linspace(1, radius)
        for dist in distances:
            floating_point = (x + (dist * cos(np.radians(angle))),  y + (dist * sin(np.radians(angle))))
            #FIXME points notation? ^
            vector.append(util.closest_pixel(floating_point))
        vectors.append(vector)

    return vectors


def extend_bresenham_line(original_image, candidate_coordinate_set, multipler = 2, border=25):

    extend = [candidate_coordinate_set[0][0] - candidate_coordinate_set[-1][0], candidate_coordinate_set[0][1] - candidate_coordinate_set[-1][1]]
    reference_point = candidate_coordinate_set[0]
    first_end = (np.array(reference_point) + (multipler * np.array(extend))).tolist()
    second_end = (np.array(reference_point) - (multipler * np.array(extend))).tolist()

    extended_can_cord_set = util.bresenham_line_points(*first_end, *second_end)
    validated_extended_can_cord_set = [point for point in extended_can_cord_set if border < point[0] < original_image.shape[0] - border and border < point[1] < original_image.shape[0] - border]
    return validated_extended_can_cord_set


def logarithmize(img):
    l_img = np.zeros_like(img)
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            if pixel == 0:
                l_img[i, j] = 255
            else:
                l_img[i, j] = round(255*-log(pixel/255))
    #util.show_img(l_img)
    return l_img


def calculate_center(img):
    l_img = logarithmize(img)
    candidate1 = util.center_of_mass(l_img)[::-1]
    candidate2 = util.center_of_mass(util.negative(l_img))[::-1]
    #logger.debug("Candidates are: " + str(candidate1) + " and " + str(candidate2))
    candidate_coordinate_set = extend_bresenham_line(img, util.bresenham_line_points(*candidate1, *candidate2), border=50)
    #logger.debug("Number of candidates: " + str(len(candidate_coordinate_set)))

    best_candidate = None
    min_variability = None
    values = []
    cords = []
    for candidate_coords in candidate_coordinate_set:
        radii_coordinates = get_radii_coordinates(img, candidate_coords, num_samples=512, offset=0)
        radii = [[img[xy] for xy in radius_coords]
                 for radius_coords in radii_coordinates]

        #current_variability = vector_variability(radii)
        current_variability = max([(sum(radius) / len(radius)) for radius in radii])

        values.append(current_variability)
        cords.append(candidate_coords)

        if best_candidate is None or current_variability > min_variability:
            min_variability = current_variability
            best_candidate = candidate_coords

    return candidate_coordinate_set, candidate1, candidate2, best_candidate #FIXME remove redundant returns


#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
def vector_variability(vectors=[]):
    v = np.asmatrix(vectors)
    pairwise_cosine_similarity_matrix = cosine_distances(v)
    flattened = np.array([], dtype=np.float64)
    for i, row in enumerate(pairwise_cosine_similarity_matrix):
        flattened = np.concatenate((flattened, row[i+1:]), axis=0)

    return np.median(flattened)


def center_visualization(img, candidates, candidate1, candidate2, center):
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    vectors = get_radii_coordinates(img, center, num_samples=80, offset=0)

    for vector in vectors:
        for pixel in vector:
            img[pixel] = [0, 200, 255]

    for candidate in candidates:
        img[candidate] = [0, 255, 0]

    cv.circle(img, center[::-1], 3, (255, 0, 0), thickness=-1)
    cv.circle(img, candidate1, 2, (255, 255, 255), thickness=-1)
    cv.circle(img, candidate2, 2, (0, 0, 0), thickness=-1)
    return img


#FIXME remove
def test(dirname):
    file_names = [fn for fn in glob.glob(dirname + "*.png")]
    for im_name in file_names:
        original_image = cv.imread(im_name, cv.IMREAD_GRAYSCALE)

        candidate_coordinate_set, candidate1, candidate2, best_candidate = calculate_center(original_image)
        result = center_visualization(original_image, candidate_coordinate_set, candidate1[::-1], candidate2[::-1], best_candidate)

        color_img = cv.cvtColor(original_image, cv.COLOR_GRAY2BGR)
        for i in candidate_coordinate_set:
            color_img[i] = [0, 255, 0]

        from util import show_img
        show_img(result)


def main(dirname="./data/"):
    file_names = [fn for fn in glob.glob(dirname + "*.png")]
    #labeled_image_names = pd.read_csv("reflex.csv").iloc[:, 0].str.slice(7, -4).values
    for im_name in file_names[::-1]:
        print(im_name)
        img = cv.imread(im_name, cv.IMREAD_GRAYSCALE)
        logger.debug("Image " + im_name + " read successfully")
        candidate_coordinate_set, candidate1, candidate2, best_candidate = calculate_center(img)
        print(best_candidate)
        data = {"image_name": im_name[len(dirname):], "x": best_candidate[1], "y": best_candidate[0]}
        util.write_to_csv("centers_ac.csv", data)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: <python3 find_center.py image_directory_name>")
        exit(-1)
    dirname = sys.argv[1]
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    main(dirname)
    #test('./center_data/')

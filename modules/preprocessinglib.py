import cv2 as cv
import numpy as np
import operator
import logging
import glob
import ntpath
import sys
import ntpath

import numpy as np
import cv2 as cv
import pandas as pd

from math import cos, sin, log
from sklearn.metrics.pairwise import cosine_distances

import util
from util import radial_angle

def draw_circle(x0, y0, r):
    x, y, p = [0, r, 1-r]
    L = []
    L.append((x, y))

    for x in range(int(r)):
        if p < 0:
            p = p + 2 * x + 3
        else:
            y -= 1
            p = p + 2 * x + 3 - 2 * y

        L.append((x, y))

        if x >= y:
            break

    N = L[:]
    for i in L:
        N.append((i[1], i[0]))

    L = N[:]
    for i in N:
        L.append((-i[0], i[1]))
        L.append((i[0], -i[1]))
        L.append((-i[0], -i[1]))

    N = []
    for i in L:
        N.append((x0+i[0], y0+i[1]))

    return N


def downscale_gray_image(img, tolerance=25, kernel=3, iterations=1, skip=True):

    y, x = img.shape
    step = kernel if skip else 1
    for _ in range(iterations):
        for xi in range(0, x - kernel + 1, step):
            for yi in range(0, y - kernel + 1, step):
                block = img[yi: yi + kernel, xi: xi + kernel]
                if block.max() - block.min() <= tolerance:
                    img[yi: yi + kernel, xi: xi + kernel] = block.max()
    return img


def logarithmize(img):
    l_img = np.zeros_like(img)
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            if pixel == 0:
                l_img[i, j] = 255
            else:
                l_img[i, j] = round(255*-log(pixel/255))
    return l_img

def sort_dict(dict, by_value=False, sort_reversed=False):
    result = sorted(dict.items(), key=operator.itemgetter(int(by_value)))
    return result[::-1] if sort_reversed else result


def extend_bresenham_line(original_image, candidate_coordinate_set, multipler = 2, border=25):

    extend = [candidate_coordinate_set[0][0] - candidate_coordinate_set[-1][0], candidate_coordinate_set[0][1] - candidate_coordinate_set[-1][1]]
    reference_point = candidate_coordinate_set[0]
    first_end = (np.array(reference_point) + (multipler * np.array(extend))).tolist()
    second_end = (np.array(reference_point) - (multipler * np.array(extend))).tolist()

    extended_can_cord_set = util.bresenham_line_points(*first_end, *second_end)
    validated_extended_can_cord_set = [point for point in extended_can_cord_set if border < point[0] < original_image.shape[0] - border and border < point[1] < original_image.shape[0] - border]
    return validated_extended_can_cord_set


'''img must be grayscale. 
num_samples in number of circles drawn. 
angle is the width of the detected ray
center must be in (x,y) format
Returns angle of ray, where 0 is at North, angle increases clockwise. When no ray detected, returns None'''
def find_ray_angle(img, center):

    # Wyznaczanie promienia na podstawie najlepszego kandydata
    radii_coordinates = get_radii_coordinates(img, center, num_samples=512, offset=0)
    radii_pixels = [[img[xy] for xy in radius_coords] for radius_coords in radii_coordinates]
    v_means = np.array([(sum(radius) / len(radius)) for radius in radii_pixels])

    mean, std = np.mean(v_means), np.std(v_means)
    v_means -= mean

    best_ray_i, _ = max(enumerate(radii_pixels), key=lambda x: np.mean(x[1]))
    ray_angle = None
    if v_means[best_ray_i] > 4 * std:
        ray_angle = util.radial_angle(center, radii_coordinates[best_ray_i][-1][::-1])

    return ray_angle


def calculate_center(img, padding=10):
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
        radii_pixels = [[img[xy] for xy in radius_coords]
                 for radius_coords in radii_coordinates]

        #current_variability = vector_variability(radii)
        x, y = candidate_coords

        # TODO: Wyznaczyć odpowiadnią wielkość powierzchni dookoła punktu centralnego
        #       Na podstawie ktorej będziemy liczyć wartość średnią i sprawdzać jak mocno
        #       Odbiega ona od ustalonego optimum w metodzie oceniania kandydatów na środek
        border_size = int(img.shape[0] / 512)

        # TODO: Pole brane pod uwagę powinno kształtem odwzorowywać koło, a nie kwadrat.
        #       Powinno poprawić to wynik, gdyż będzie zgodne z samą naturą zdjeć.
        fileds = img[x - border_size : x + border_size, y - border_size : y + border_size].flatten()
        mean_center_pixel_border = sum(fileds) / len(fileds)

        # TODO: Czy dlugosc wektora jest zawsze taka sama? Dlaczego?
        padding_value = round(padding * len(radii_pixels[0]) / 100)

        # TODO: Ustalić kolor, które będzie optimum, od które będzie nam ważyć ilość punktow dla kandydata
        #       Na podstawie obserwacji zdjęć punkt centralny zdjęcia zawsze znajduje się albo na białym tle albo na czarnym.
        #       Wszystko obok tego punktu jest szare. Więc jeżeli na naszej lini leży kandydat, który znajduje się na czarnym badź
        #       białym tle to dostaje on więcej "punktów". Szare tło jest powszechne i nie dostaje za nie dodatkowych puntów.
        #       Oczywiście branie jednego pixela nie jest reprezentatywne dlatego pod uwagę bierze się obszar dookoła środka,
        #       którego rozmiar powinien zostać wyznaczony kilka linii wyżej przez zmienną "border_size"
        current_variability = max([(sum(radius[padding_value:]) / len(radius[padding_value:])) for radius in radii_pixels]) * (abs(90 - (mean_center_pixel_border)) / 255)

        values.append(current_variability)
        cords.append(candidate_coords)

        if best_candidate is None or current_variability > min_variability:
            min_variability = current_variability
            best_candidate = candidate_coords

    return candidate_coordinate_set, candidate1, candidate2, best_candidate #FIXME remove redundant returns


'''Returns vector of radii, starting with north and going clockwise.
Returns numpy-style (y,x) format'''
def get_radii_coordinates(img, center, num_samples=20, offset=0):

    x, y = center
    if not 0 < x < img.shape[1] or not 0 < y < img.shape[0]:
        return []

    angles = np.linspace(0 + offset, 360 + offset, num_samples + 1)
    angles = angles + 180
    radius = np.min([img.shape[1] - x - 1, x, img.shape[0] - y - 1, y])

    vectors = []
    for angle in angles:
        vector = []
        for dist in np.linspace(1, radius):
            floating_point = (x + (dist * cos(np.radians(angle))),  y - (dist * sin(np.radians(angle))))
            vector.append(util.closest_pixel(floating_point))
        vectors.append(vector)

    return vectors


def t_ray_detector(): #TODO

    DATA_PATH = "/Volumes/Alice/reflex-data/reflex_img_512_inter_nearest/"
    all_examples = [fn for fn in glob.glob(DATA_PATH + "*.png")]

    for example in all_examples:
        print(example)
        image = cv.imread(example, cv.IMREAD_GRAYSCALE)
        angle = find_ray_angle(image, (255, 255), visualise=True)
        print(angle)
        #cv.imwrite("./xray_test/test_" + str(i + offset) + ".png", np.hstack((original, image)))

#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
def vector_variability(vectors=[]):
    v = np.asmatrix(vectors)
    pairwise_cosine_similarity_matrix = cosine_distances(v)
    flattened = np.array([], dtype=np.float64)
    for i, row in enumerate(pairwise_cosine_similarity_matrix):
        flattened = np.concatenate((flattened, row[i+1:]), axis=0)

    return np.median(flattened)


def center_visualization(img, candidates, candidate1, candidate2, center, angle, padding=10):
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    vectors = get_radii_coordinates(img, center, num_samples=512, offset=0)
    padding_value = round(padding * len(vectors[0]) / 100)

    for vector in vectors:
        for pixel in vector[padding_value:]:
            img[pixel] = [0, 155, 0]

    for candidate in candidates:
        img[candidate] = [0, 255, 0]

    if angle:
        ray_coords = get_radii_coordinates(img, center, 1, angle)[0]
        for coord in ray_coords:
            img[coord] = [0,0,255]

    cv.circle(img, center[::-1], 3, (255, 0, 0), thickness=-1)
    cv.circle(img, candidate1, 2, (255, 255, 255), thickness=-1)
    cv.circle(img, candidate2, 2, (0, 0, 0), thickness=-1)
    return img


#FIXME remove
def test(dirname, sourcedir, padding=10):

    testcases = [ntpath.split(fn)[1] for fn in glob.glob(dirname + "*.png")]
    allfiles = [fn for fn in glob.glob(sourcedir + "*.png")]
    all = True

    for im_name in allfiles:
        if ntpath.split(im_name)[1] in testcases or all:
            original_image = cv.imread(im_name, cv.IMREAD_GRAYSCALE)

            candidate_coordinate_set, candidate1, candidate2, center = calculate_center(original_image, padding)
            angle = find_ray_angle(original_image, center)
            result = center_visualization(original_image, candidate_coordinate_set, candidate1[::-1], candidate2[::-1], center, angle, padding)

            filename = ntpath.split(im_name)[1]
            print(filename)
            #cv.imwrite('./test_img/' + filename, result)

            from util import show_img
            show_img(result)


def main(dirname="./data/"):

    file_names = [fn for fn in glob.glob(dirname + "*.png")]
    #labeled_image_names = pd.read_csv("reflex.csv").iloc[:, 0].str.slice(7, -4).values

    for im_name in file_names[::-1]:
        print(im_name)
        img = cv.imread(im_name, cv.IMREAD_GRAYSCALE)
        #logger.debug("Image " + im_name + " read successfully")
        candidate_coordinate_set, candidate1, candidate2, center = calculate_center(img)
        angle = find_ray_angle(img, center)
        print(center)
        data = {"image_name": im_name[len(dirname):], "x": center[1], "y": center[0]}
        util.write_to_csv("centers_ac.csv", data)


if __name__ == "__main__":

    #rays = [[1, 5], [2, 10], [3, 15], [4, 16]]
    #rays.sort(key=lambda x: np.mean(x), reverse=True)
    #print(rays)

    test('/Volumes/DATA/reflex_data/best/', '/Volumes/DATA/reflex_data/reflex_img_512_inter_nearest/', padding=20) ## TODO: REMOVE
    #if len(sys.argv) < 2:
     #   print("Usage: <python3 find_center.py image_directory_name>")
     #   exit(-1)
    #dirname = sys.argv[1]
    #logging.basicConfig(level=logging.INFO)
    #logger = logging.getLogger(__name__)

    #main(dirname)
    #test('./center_data/')

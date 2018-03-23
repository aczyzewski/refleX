import numpy as np
import cv2 as cv
import os
import sys
import glob
import logging

from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from math import sin, cos

import preprocess
import util


def get_radii_coordinates(img, center, num_samples=20, offset=0):

    """
    Example:
    ------
    >>> sampling_vectors(img, (3, 3), 4, 0)
    [[(3, 3), (4, 3), (5, 3), (6, 3)],
     [(3, 3), (3, 4), (3, 5), (3, 6)],
     [(3, 3), (2, 3), (1, 3), (0, 3)],
     [(3, 3), (3, 2), (3, 1), (3, 0)]]

    """
    x, y = center
    if not 0 < x < img.shape[1] or not 0 < y < img.shape[0]:
        return []

    angles = np.linspace(0 + offset, 360 + offset, num_samples + 1)
    radius = np.min([img.shape[1] - x - 1, x, img.shape[0] - y - 1, y])

    vectors = []
    for angle in angles:
        point = (round(x + (radius * cos(np.radians(angle)))),  round(y + (radius * sin(np.radians(angle)))))
        vectors.append(bresenham_line_points(*center, *point))

    return vectors[:-1]


#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
def vector_variability(vectors=[]):
    pairwise_cosine_similarity_matrix = cosine_similarity(vectors)
    flattened = []
    for i, row in enumerate(pairwise_cosine_similarity_matrix):
        print(row)
        flattened.append(row[i+1:])
    print(np.median(flattened))
    return np.median(flattened)


def center_of_mass(img):
    m = cv.moments(img);
    if m['m00'] != 0:
        x = m['m10'] / m['m00']
        y = m['m01'] / m['m00']
    else:
        return img.shape[0]/2, img.shape[1]/2
    return x, y


def negative(img):
    return 255 - img


def bresenham_line_points(x1, y1, x2, y2):
    """
    Zwraca listę punktów, przez które przechodzić będzie prosta
    o zadanym początku i końcu

    Parametry
    ----------
    x1, y1, x2, y2 : int
        (x1, y1) - punkt poczatkowy
        (x2, y2) - punkt końcowy

    """
    # Zmienne pomocnicze
    d = dx = dy = ai = bi = xi = yi = 0
    x = x1
    y = y1
    points = []

    # Ustalenie kierunku rysowania
    xi = 1 if x < x2 else -1
    dx = abs(x1 - x2)

    # Ustalenie kierunku rysowania
    yi = 1 if y1 < y2 else -1
    dy = abs(y1 - y2)

    # Pierwszy piksel
    points.append((x, y))

    ai = -1 * abs(dy - dx) * 2
    bi = min(dx, dy) * 2
    d = bi - max(dx, dy)

    # Oś wiodąca OX
    if dx > dy:
        while x != x2:
            if d >= 0:
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                x += xi

            points.append((x, y))

    # Oś wiodąca OY
    else:
        while y != y2:
            if d >= 0:
                x += xi
                y += yi
                d += ai
            else:
                d += bi
                y += yi

            points.append((x, y))

    return points


def calculate_center(im):
    img = im.copy()
    candidate1 = center_of_mass(img)
    candidate2 = center_of_mass(negative(img))
    candidate_coordinate_set = bresenham_line_points(*candidate1, *candidate2)
    best_candidate = candidate1
    min_variability = vector_variability(get_radii_coordinates(img, best_candidate, n=20))
    for candidate_coords in candidate_coordinate_set[1:]:
        radii_coordinates = get_radii_coordinates(img, best_candidate, n=20, offset=4)
        for radius in radii_coordinates:
            for pixel in radius:
                img[pixel] = 255
        radii = [[img[xy] for xy in radius_coords]
                 for radius_coords in radii_coordinates]
        current_variability = vector_variability(radii)
        if current_variability < min_variability:
            min_variability = current_variability
            best_candidate = candidate_coords
    return img, best_candidate


#return discrete coordinates of pixels on circle of radius r, given the center of the circle
#Inspired by https://en.wikipedia.org/wiki/Midpoint_circle_algorithm, but without gaps between circles
#TODO this could be smoother (see 100x100)
def circle_points(center, r, last_layer=[]):
    logger.debug("Calculating circle for r = " + str(r))
    x0 = center[0]
    y0 = center[1]
    points = []
    f = 1 - r
    ddf_x = 1
    ddf_y = -2 * r
    x = 0
    y = r
    points.append((x0, y0 + r))
    points.append((x0, y0 - r))
    points.append((x0 + r, y0))
    points.append((x0 - r, y0))

    while x < y:
        if f >= 0:
            y -= 1
            ddf_y += 2
            f += ddf_y
        x += 1
        ddf_x += 2
        f += ddf_x
        coords = ((x0 + x, y0 + y),
                  (x0 - x, y0 + y),
                  (x0 + x, y0 - y),
                  (x0 - x, y0 - y),
                  (x0 + y, y0 + x),
                  (x0 - y, y0 + x),
                  (x0 + y, y0 - x),
                  (x0 - y, y0 - x))
        shifts = ((-1, -1), (1, -1), (-1, 1), (1, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))
        for coord, shift in zip(coords, shifts):
            points.append(coord)
            potentially_unfilled = tuple(map(lambda a, b: a+b, coord, shift))
            if potentially_unfilled not in last_layer:
                points.append(potentially_unfilled)

    logger.debug(points)
    return points


def get_layers(img, center):
    layers = []
    r_max = min(center[0], center[1], img.shape[0]-center[1], img.shape[1]-center[0])
    for r in range(r_max):
        if r != 0:
            layers.append(circle_points(center, r, layers[-1]))
        else:
            layers.append([center])
    return layers


def color_layers(img, layers, color_alpha=0.1, gamma=0):
    r_max = len(layers)
    color = np.zeros_like(img)
    for r, l in enumerate(layers):
        for coord in l:
            color[coord] = util.hsv2rgb(util.linear_interpolation(r, 0, r_max, 0, 360), 1, 1)
    return cv.addWeighted(color, color_alpha, img, 1-color_alpha, gamma)


def apply_aggregate_fcn(img, stat_fcn, irrelevant, layers=[]):
    values = [[img[coord] for coord in coords if not irrelevant[coord]] for coords in layers]
    aggregated = [stat_fcn(layer) if layer else None for layer in values]
    if aggregated[0] is None:
        aggregated[0] = 0 #FIXME dunno what to do if first result is empty
    for i in range(len(aggregated)):
        if aggregated[i] is None:
            aggregated[i] = aggregated[i - 1]
    return aggregated
#TODO taking last non-empty result might not be the best idea


def parse_command_line_options():
    usage = "Usage: radial_compressor.py [input_directory='./data')] [-s compression statistics separated by spaces]"
    dir_name = "./data/" if len(sys.argv) == 1 or sys.argv[1] == "-s" else sys.argv[1]
    if dir_name[-1] != '/':
        dir_name += '/'
    file_names = [fn for fn in glob.glob(dir_name + "*.png")]
    chosen_stats = sys.argv[sys.argv.index("-s") + 1:] if "-s" in sys.argv else list(stat_names.keys())
    for s in chosen_stats:
        if s not in stat_names.keys():
            logger.error(usage)
            logger.error("Available statistics: ", stat_names.keys())
            exit(0)
    logger.info("Processing " + str(len(file_names)) + " png files from " + dir_name)
    compressed_dir_name = "./1d_" + dir_name[2:]
    if not os.path.exists(compressed_dir_name):
        os.makedirs(compressed_dir_name)
    logger.info("Saving results in " + compressed_dir_name)

    if logging.getLogger().isEnabledFor(logging.INFO):
        info_dir_name = "./info_" + dir_name[2:]
        if not os.path.exists(info_dir_name):
            os.makedirs(info_dir_name)

    col_width = max([len(i) for i in file_names]) + 3
    return file_names, chosen_stats, col_width


def main():
    file_names, chosen_stats, col_width = parse_command_line_options()
    for im_name in file_names:
        original_image = cv.imread(im_name, cv.IMREAD_GRAYSCALE)
        #im = preprocess.narrow_gaps(original_image) #FIXME
        im = original_image #FIXME ^
        irrelevant = preprocess.radial_mark_irrelevant(im)

        center_visualization, center = calculate_center(im)
        logger.debug(im_name.ljust(col_width) + str(center))

        layers = get_layers(im, center)
        compressed_1d_images = []
        for s in chosen_stats:
            vector = apply_aggregate_fcn(im, stat_names[s], irrelevant, layers)
            compressed_1d_images.append(vector)
            vector = np.array(vector)
            cv.imwrite("./1d_" + im_name[2:-4] + s + ".png", vector)

        if logging.getLogger().isEnabledFor(logging.INFO):
            fig, axs = plt.subplots(nrows=int(np.ceil((len(compressed_1d_images)+3)/3)), ncols=3)
            axs[0][0].imshow(original_image)
            axs[0][0].set_title("Original")
            axs[0][1].imshow(irrelevant)
            axs[0][1].set_title("Marked irrelevant")
            axs[0][2].imshow(center_visualization)
            axs[0][2].set_title("Detected center")
            for i,v in enumerate(compressed_1d_images):
                axs[1+i//3][i%3].imshow(np.vstack(15 * (v,)))
                axs[1+i//3][i%3].set_title(chosen_stats[i])
            plt.axis('off')
            plt.savefig("./info_" + im_name[2:-4] + "_info.png")
            plt.close()

    logger.info("Finished")
    cv.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    stat_names = {"max": max,
                  "min": min,
                  "mean": np.mean,
                  "median": np.median,
                  "var": np.var
                  }
    main()

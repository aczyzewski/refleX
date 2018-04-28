import argparse
import glob
import logging
import os

import cv2 as cv
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


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


def process_image(center_dict, computed_centers, col_width, im_name_with_dir, chosen_stat_names, compressed_dirname):

    im_name = im_name_with_dir[im_name_with_dir.rfind('/')+1:]
    if im_name not in computed_centers:
        return

    logger.debug("Processing " + im_name)
    original_image = cv.imread(im_name_with_dir, cv.IMREAD_GRAYSCALE)
    logger.debug("Image " + im_name + " read successfully")

    im = original_image
    irrelevant = np.zeros_like(im) #FIXME
    logger.debug(im_name + " preprocessed successfully")
    center = (center_dict[im_name]['x'], center_dict[im_name]['y'])
    logger.info("Center in: " + str(center))
    logger.debug("Found center: ")
    logger.debug(im_name.ljust(col_width) + str(center))

    layers = get_layers(im, center)
    compressed_1d_images = []
    for s in chosen_stat_names:
        vector = apply_aggregate_fcn(im, stat_functions[s], irrelevant, layers)
        compressed_1d_images.append(vector)
        vector = np.array(vector)
        cv.imwrite(compressed_dirname + s + "/" + im_name, vector)


def configure_parser(parser): #TODO allow numeric percentile arg?
    parser.add_argument("image_dirname", help="Path to directory containing images for compression.")
    parser.add_argument("centers_csv_filename",
                        help="Filename of csv containing image filenames and x and y coordinates the image centers")
    parser.add_argument("-var", "--var", action="store_true", help="Use variance for compression.")
    parser.add_argument("-mean", "--mean", action="store_true", help="Use mean for compression.")
    parser.add_argument("-median", "--median", action="store_true", help="Use median for compression.")
    parser.add_argument("-min", "--min", action="store_true", help="Use min for compression.")
    parser.add_argument("-max", "--max", action="store_true", help="Use max for compression.")
    parser.add_argument("-5th", "--5th", action="store_true", help="Use 5th percentile for compression.")
    parser.add_argument("-95th", "--95th", action="store_true", help="Use 95th percentile for compression.")
    parser.add_argument("-allstats", "--allstats", action="store_true",
                        help="Run all methods of compression (" + " ".join([str(s) for s in stat_functions.keys()]) + ").")
    parser.add_argument("-nj", "--n_jobs", type=int, nargs="?", const=1, help="Number of threads. Default 1 thread.")


def main():
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    args = parser.parse_args()

    file_names = [fn for fn in glob.glob(args.image_dirname + "*.png")]
    logger.info("Processing " + str(len(file_names)) + " png files from " + args.image_dirname)
    compressed_dir_name = args.image_dirname[:-1] + "_compressed/"
    if not os.path.exists(compressed_dir_name):
        os.makedirs(compressed_dir_name)
    logger.info("Saving results in " + compressed_dir_name)
    col_width = max([len(i) for i in file_names]) + 3

    chosen_stat_names = []
    if args.allstats:
        chosen_stat_names = list(stat_functions.keys())
    else:
        for stat_name in stat_functions.keys():
            if args.stat_name:
                chosen_stat_names.append(stat_name)
    if len(chosen_stat_names) == 0:
        print("Must choose at least one compression statistic")
        parser.print_help()
        exit(-1)

    for stat_name in chosen_stat_names:
        os.makedirs(compressed_dir_name + stat_name)

    center_dict = pd.read_csv(args.centers_csv_filename).set_index('image_name').to_dict('index')
    computed_centers = list(center_dict.keys())

    Parallel(n_jobs=args.n_jobs)(delayed(process_image)
                       (center_dict, computed_centers, col_width, im_name, chosen_stat_names, compressed_dir_name)
                       for im_name in file_names)

    logger.info("Finished")
    cv.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    stat_functions = {
        "max": np.nanmax,
        "95th_percentile": lambda x: np.nanpercentile(x, 95),
        "min": np.nanmin,
        "5th_percentile": lambda x: np.nanpercentile(x, 5),
        "mean": np.nanmean,
        "median": np.nanmedian,
        "var": np.nanvar
    }

    main()

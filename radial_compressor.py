import numpy as np
import cv2 as cv
import pandas as pd
import os
import sys
import glob
import logging
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

import preprocess
import util
import find_center


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
    dir_name = "./center_data/" if len(sys.argv) == 1 or sys.argv[1] == "-s" else sys.argv[1]
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


def process_image(center_dict, computed_centers, col_width, im_name, chosen_stats):

    if im_name[im_name.rfind('/') + 1:] not in computed_centers:
        return

    logger.debug("Processing " + im_name)
    original_image = cv.imread(im_name, cv.IMREAD_GRAYSCALE)
    logger.debug("Image " + im_name + " read successfully")

    # im = preprocess.narrow_gaps(original_image) #FIXME
    im = original_image  # FIXME ^
    irrelevant = preprocess.radial_mark_irrelevant(im)
    logger.debug(im_name + " preprocessed successfully")

    center = (center_dict[im_name[39:]]['x'], center_dict[im_name[39:]]['y'])
    logger.info("Center in: " + str(center))
    logger.debug("Found center: ")
    logger.debug(im_name.ljust(col_width) + str(center))

    layers = get_layers(im, center)
    compressed_1d_images = []
    for s in chosen_stats:
        vector = apply_aggregate_fcn(im, stat_names[s], irrelevant, layers)
        compressed_1d_images.append(vector)
        vector = np.array(vector)
        cv.imwrite("./1d_" + im_name[2:-4] + s + ".png", vector)

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        fig, axs = plt.subplots(nrows=int(np.ceil((len(compressed_1d_images) + 3) / 3)), ncols=3, figsize=(20, 10))
        axs[0][0].imshow(original_image)
        axs[0][0].set_title("Original")
        axs[0][1].imshow(irrelevant)
        axs[0][1].set_title("Marked irrelevant")
        axs[0][2].imshow(find_center.center_visualization(im, [], None, None, center[::-1]))
        axs[0][2].set_title("Detected center")
        for i, v in enumerate(compressed_1d_images):
            axs[1 + i // 3][i % 3].imshow(np.vstack(15 * (v,)))
            axs[1 + i // 3][i % 3].set_title(chosen_stats[i])
        plt.axis('off')
        plt.savefig("./info_" + im_name[2:-4] + "_info.png")
        plt.close()


def main():
    file_names, chosen_stats, col_width = parse_command_line_options()
    center_dict = pd.read_csv("centers.csv").set_index('image_name').to_dict('index')
    computed_centers = list(center_dict.keys())

    Parallel(n_jobs=8)(delayed(process_image)
                       (center_dict, computed_centers, col_width, im_name, chosen_stats)
                       for im_name in file_names)

    logger.info("Finished")
    cv.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    stat_names = {"max": lambda x: np.nanpercentile(x, 95),
                  "min": lambda x: np.nanpercentile(x, 5),
                  "mean": np.nanmean,
                  "median": np.nanmedian,
                  "var": np.nanvar
                  }
    main()

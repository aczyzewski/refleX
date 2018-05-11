import argparse
from modules.preprocessinglib import calculate_center, find_ray_angle
from modules.radial_compressor import process_image
from modules.rawdatalib import RawDataFile
import os

def save_csv():
    pass

def init_converting_files(dir):

    files = [dir + file for file in os.listdir(dir) if file.endswith('.npy.bz2')]
    for file in files:
        raw_array = RawDataFile(file)
        numpy_array = raw_array.npy_to_img(size=512)
        _, _, _, center = calculate_center(numpy_array)
        angle = find_ray_angle(numpy_array, center)
        save_csv()







if __name__ == '__main__':

    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()
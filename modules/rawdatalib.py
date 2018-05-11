import numpy as np
import bz2
import ntpath
import cv2 as cv
import os
import logging

#TODO: Documentation!

#TODO: Logger
# https://stackoverflow.com/questions/34940302/should-i-use-logging-module-or-logging-class
# https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class RawDataFile():

    def __init__(self, path, data_ext='.npy.bz2', info_ext='.info', load_data=True):
        self.__data_ext = data_ext
        self.__info_ext = info_ext
        self.__parse_path(path)
        self.clear_data()
        self.data = None

        if load_data:
            self.load_data()

    def clear_data(self):
        self.data = np.array([])
        self.info = {}

    def load_data(self):
        if os.path.isfile(self.fullpath + self.__info_ext):
            self.load_info()

        if os.path.isfile(self.fullpath + self.__data_ext):
            self.load_npyarray()

    def set_path(self, path):
        self.__parse_path(path)
        self.load_data()

    def __str__(self):
        return self.fullpath + self.__data_ext

    def __parse_path(self, path):
        dir, filename = ntpath.split(path)
        self.dir, self.filename = dir + '/', filename[:-len(self.__data_ext)]
        self.fullpath = self.dir + self.filename

    def load_info(self):
        try:
            with open(self.fullpath + self.__info_ext, 'r') as file:
                for line in file.readlines():
                    parameters = line.split()
                    self.info[parameters[0]] = ' '.join(parameters[1:])
        except FileNotFoundError:
            logging.debug(f"File error: {self.fullpath + self.__info_ext}!")

    def load_npyarray(self):
        try:
            self.data = np.load(bz2.BZ2File(self.fullpath + self.__data_ext))
        except:
            logging.debug(f"Array error: {self.fullpath + self.__info_ext}!")

    def __autoscale_legacy(self, data):
        n = data.shape[0] // 12
        s = np.nansum(data[2 * n:3 * n]) + np.nansum(data[9 * n:10 * n])
        mn = 1
        mx = 5 * s / (2 * n * data.shape[1])
        return (mn, mx)

    def to_img(self, size=None, delete_grid=True, padding=25, interpolation=cv.INTER_NEAREST):

        if type(self.data) is np.ndarray:
            data = self.data.copy()
            data = data[padding:-padding, padding:-padding]
            mn, mx = self.__autoscale_legacy(data)

            if delete_grid:
                local_data = data.copy()
                local_data += 1
                local_data = np.nan_to_num(data)
                lookuptable = np.zeros(data.shape)
                lookuptable[np.where(local_data == 0)] = 1
                cols_to_del, rows_to_del = [], []

                for i in range(data.shape[0]):
                    if np.all(lookuptable[i, :]):
                        rows_to_del.append(i)

                for i in range(data.shape[1]):
                    if np.all(lookuptable[:, i]):
                        cols_to_del.append(i)

                data = np.delete(data, rows_to_del, 0)
                data = np.delete(data, cols_to_del, 1)

            data = np.nan_to_num(data)
            data[np.where(data > mx)] = mx
            data /= mx
            data *= 255

            if size:
                data = cv.resize(data, (size, size), interpolation=interpolation)

            return 255 - data.astype(np.uint8)
        return None


def convert_files(input_dir, output_dir, size=None, input_ext = '.npy.bz2', output_ext = '.png', overwrite=False):
    files = [file for file in os.listdir(input_dir) if file.endswith(input_ext) and not file.startswith('.')]

    if not overwrite:
        existing_files = [file[:-len(output_ext)] for file in os.listdir(output_dir) if file.endswith(output_ext) and not file.startswith('.')]
        files = [file for file in files if file[:-len(input_ext)] not in existing_files]

    for idx, file in enumerate(files):
        try:
            raw_data = RawDataFile(input_dir + file)
            image = raw_data.npy_to_img()
            result = cv.resize(image, (size, size), interpolation=cv.INTER_NEAREST) if type(size) is int else image
            size_info = f'.{size}x{size}' if type(size) is int else ""
            cv.imwrite(f'{output_dir}{raw_data.filename}{size_info}.png', result)
        except:
            print("Cannot convert file: " + file)

def parameters_in_file(DIR, filename, parameters):

    with open(DIR + filename + ".info") as file:
        data = file.read()
        if all([param in data for param in parameters]):
            return True

    return False

def get_files_without_params(files, params, debug=False):
    results = []
    for file in files:
        if not parameters_in_file(file, params):
            results.append(file)
            if debug:
                print(f"Set of parameters: {params} not found in {file}.info")
    return results

def get_params_from_file(filename, params):

    if type(params) is not list: params = [params]
    results = {key: None for key in params}

    with open(filename) as file:
        data = file.read().split()
        for param in params:
            if param in data:
                results[param] = data[data.index(param) + 1]

    return results
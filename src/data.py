import os
import math
import warnings
from collections import Counter
from typing import Any, List, Union, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from skmultilearn.model_selection import iterative_train_test_split

from src.utils import splitpath


def calculate_mean_std(images: List[str]) -> Tuple[float, float]:
    """ Calculates mean/std of given images (grayscale)
        Note: this method is quite inefficient. """

    # TODO: Tests
    warnings.warn('[calculate_mean_std] Warning: Untested function!')

    def _calculate_mean(counter: Counter) -> float:
        sum_of_numbers = sum(number * count
                             for number, count in counter.items())
        count = sum(count for n, count in counter.items())
        mean = sum_of_numbers / count
        return mean / 255

    def _calculate_std(counter: Counter, mean: float) -> float:
        total_squares = sum(number * number * count
                            for number, count in counter.items())
        count = sum(count for n, count in counter.items())
        mean_of_squares = total_squares / count
        variance = mean_of_squares - mean * mean
        std_dev = math.sqrt(variance)
        return std_dev / 255

    counter = Counter()
    for imgpath in tqdm(images):
        values = np.array(Image.open(imgpath))
        counter.update(values.flatten().tolist())

    mean = _calculate_mean(counter)
    std = _calculate_std(counter, mean)

    return mean, std


def train_val_test_split(X: Any, y: Any, validation_size: float = 0.25,
                         test_size: float = 0.25) -> List[Any]:
    """ Splits input data into train/val/test subsets """

    X_train, y_train, X_test, y_test = iterative_train_test_split(
        X, y, test_size=test_size)

    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X_train, y_train, test_size=validation_size)

    return X_train, y_train, X_val, y_val, X_test, y_test


def resize_images(image: Union[str, List[str]], size: Union[int, List[int]],
                  output: str = 'resized', method: Any = Image.NEAREST,
                  colorspace: str = 'L', overwrite: bool = False) -> None:
    """ Loads, resizes and saves the image(s) in the given location """

    def _convert_to_list(x): return [x] if not isinstance(x, list) else x

    # Create directory tree
    for size in _convert_to_list(size):
        os.makedirs(os.path.join(output, str(size)), exist_ok=True)

    for imagepath in tqdm(_convert_to_list(image)):
        image = Image.open(imagepath).convert(colorspace)
        basedirectory, name, ext = splitpath(imagepath)
        assert image.size[0] == image.size[1], \
            f'{imagepath}: Image width =/= image height!'

        for size in _convert_to_list(size):

            # Skip file if exists
            output_file_path = os.path.join(output, str(size), f'{name}{ext}')
            if not overwrite and os.path.isfile(output_file_path):
                continue

            temp = image.resize((size, size), method)
            temp.save(output_file_path)

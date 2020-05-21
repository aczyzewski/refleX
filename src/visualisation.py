from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Custom type aliases
Shape2D = Tuple[int, int]


def make_grid(images: List[str], shape: Shape2D, cmap: str = 'gray',
              figsize: Shape2D = (16, 16)) -> None:
    """ Makes grid of the given shape of the given image list.
        The images should be privded as list of paths """

    assert len(shape) == 2, 'Invalid grid shape!'
    x, y = shape

    assert len(images) <= x * y, 'The number of photos exceeded the grid size.'
    assert x > 0 and y > 0, 'The number of rows/cols has be greater than zero!)'

    plt.figure(figsize=figsize)
    for idx, image in enumerate(images, start=1):
        ax = plt.subplot(x, y, idx)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        img = np.array(Image.open(image))
        plt.imshow(img, cmap=cmap)

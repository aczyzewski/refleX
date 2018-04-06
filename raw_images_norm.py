import bz2
import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

def autoscale_legacy(data):
    n = data.shape[0] // 12                                     # Changed from '/' to '//'
    s = np.nansum(data[2*n:3*n]) + np.nansum(data[9*n:10*n])
    mn = 1
    mx = 5*s/(2*n*data.shape[1])
    return (mn, mx)


def main():
    DIR = './pd_image_dumps/'
    data_ext = '.npy.bz2'
    info_ext = '.info'

    files = [file[:-8] for file in os.listdir(DIR) if file.endswith(data_ext)]

    for file in files:
        with bz2.BZ2File(DIR + file + data_ext) as fh:
            data = np.load(fh)
            mn, mx = autoscale_legacy(data)
            data = np.nan_to_num(data)
            data[np.where(data > mx)] = mx

            img_norm = data.copy()
            img_norm /= mx
            img_norm *= 255
            img_color = cv2.cvtColor(255 - img_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)

            cv2.imwrite('./raw_norm/' + file + '.png', img_color)
            print('Saved: ' + file)

if __name__ == '__main__':
    main()
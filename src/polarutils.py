import numpy as np
import cv2
from preprocessinglib import calculate_center

def img2polar(img, center, method='min'):
    
    def _minimal():
        return min(min(center[0], img.shape[0]-center[0]), min(center[1], img.shape[1]-center[1]))
    def _maximal():
        return np.sqrt(
            min(center[0], img.shape[0]-center[0])**2
            + min(center[1], img.shape[1]-center[1])**2
        )
    
    max_radius = _minimal() if method == 'min' else _maximal()
    polar_image = cv2.linearPolar(img, center, max_radius, cv2.INTER_NEAREST)#TODO
    return polar_image.astype(np.uint8)


def polarize(img_filename, input_directory, output_directory, center_df, method='min'):
    original = cv2.imread(f'{input_directory}{img_filename}', 1)
    #img_filename = full_filename[:full_filename.rfind("_")] + '.png'
    if img_filename in center_df.index:
        center = (center_df.loc[img_filename].loc['y'], center_df.loc[img_filename].loc['x']) # y,x
    else:
        print('CENTER NOT FOUND: ', img_filename)
        center = calculate_center(original)
    polar = img2polar(original, center, method=method)
    #_, ax = plt.subplots(1, 2)
    #ax[1].imshow(polar)
    #ax[0].imshow(original)
    #plt.show()
    cv2.imwrite(output_directory+img_filename, polar)
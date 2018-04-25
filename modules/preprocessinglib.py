#TODO clean up

import cv2 as cv
import numpy as np
import glob
import operator

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


def sort_dict(dict, by_value=False, sort_reversed=False):
    result = sorted(dict.items(), key=operator.itemgetter(int(by_value)))
    return result[::-1] if sort_reversed else result


'''img must be grayscale. 
num_samples in number of circles drawn. 
angle is the width of the detected ray
center must be in (x,y) format
Returns angle of ray, where 0 is at North, angle increases clockwise. When no ray detected, returns None'''
def find_ray_angle(img, center, angle=30, num_samples=6, visualise=False):

    x_center, y_center = center
    radius = np.min([img.shape[1] - x_center - 1, x_center, img.shape[0] - y_center - 1, y_center])

    threshold = 245  # FIXME MAGIC NUMBER !

    # Sample pixels using circles
    sample_radii = [int(i) for i in np.linspace(10, radius - 1, num_samples + 1)]
    circles = []
    for i in range(len(sample_radii)):
        circles.append(draw_circle(x_center, y_center, sample_radii[i])[::-1])

    # Ray detecting
    ref_points = circles[-1]
    ray_angles = [] #array of vectors of points, which form a ray

    circle_points = [p for c in circles for p in c]

    for ref_point in ref_points:
        n_inter = 0
        line = util.bresenham_line_points(x_center, y_center, ref_point[0], ref_point[1])
        intersected_circle_points = set.intersection(set(line), set(circle_points))

        for point in intersected_circle_points:
            if img[point[::-1]] > threshold:
                n_inter += 1

        if len(circles) == n_inter:
            ray_angles.append(radial_angle((x_center, y_center), (ref_point[0], ref_point[1])))

    ray_angles.sort()
    print(ray_angles)

    # Find segment with largest number of marked rays
    max_rays_in_segment = 0
    best_angle = None
    for angle_start in range(0, 365, angle // 2):
        n_rays = 0
        to_delete = []
        for i, _angle in enumerate(ray_angles):
            if _angle < angle_start + angle/2:
                to_delete.append(i)
            if _angle >= angle_start and _angle <= angle_start + angle_start:
                n_rays += 1
        if n_rays > max_rays_in_segment:
            max_rays_in_segment = n_rays
            best_angle = angle_start+angle/2
        for i in to_delete[::-1]:
            del ray_angles[i]

    return best_angle


def t_ray_detector(): #TODO

    DATA_PATH = "/Volumes/Alice/reflex-data/reflex_img_512_inter_nearest/"
    all_examples = [fn for fn in glob.glob(DATA_PATH + "*.png")]

    for example in all_examples:
        print(example)
        image = cv.imread(example, cv.IMREAD_GRAYSCALE)
        angle = find_ray_angle(image, (255, 255), visualise=True)
        print(angle)
        #cv.imwrite("./xray_test/test_" + str(i + offset) + ".png", np.hstack((original, image)))


if __name__=="__main__":
    t_ray_detector()
    #t_radial_angle()
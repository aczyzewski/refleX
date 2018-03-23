import cv2 as cv
import numpy as np
import glob
from math import sin, cos, pow, atan2
import operator

def normalize_gray_image(img, base=8):
    base = pow(2, base) - 1
    float_array = np.array(img, dtype=np.float64)
    float_array -= float_array.min()
    float_array *= float(base) / float_array.max()
    return np.array(np.around(float_array), dtype=np.uint8)

def show_img(image, width=600, height=600):
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.resizeWindow('image', width, height)
    cv.moveWindow('image', 10, 10)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

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
                block = img[yi: yi + kernel, xi : xi + kernel]
                if block.max() - block.min() <= tolerance:
                    img[yi: yi + kernel, xi: xi + kernel] = block.max()
    return img

def calculate_centers(image):
    # Temporarty solution
    return [x // 2 for x in image.shape[::-1]]

def sort_dict(dict, byValue=False, reversed=False):
    result = sorted(dict.items(), key=operator.itemgetter(int(byValue)))
    return result[::-1] if reversed else result

def sort_circle_points(points):
    points.sort(key=lambda c: atan2(c[0], c[1]))
    return points

def shift_list(seq, n=0):
    a = n % len(seq)
    return seq[-a:] + seq[:-a]

def ray_detector(img, alpha=30, num_samples=6, visualise=False):

    # Prepare image
    grayscale = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
    grayscale = downscale_gray_image(grayscale, kernel=4)
    grayscale = normalize_gray_image(grayscale)

    # Calculate important values
    radius = min(grayscale.shape[0], grayscale.shape[1]) // 2
    x_center, y_center = calculate_centers(grayscale)
    threshold = 245  # MAGIC NUMBER !

    # Sample pixels using circles
    sample_radii = [int(i) for i in np.linspace(10, radius - 1, num_samples + 1)]

    circles = []
    for i in range(len(sample_radii)):
        circles.append(draw_circle(x_center, y_center, sample_radii[i])[::-1])

    # Ray detecting
    ref_points = circles[-1]
    ray_lines = []

    for ref_point in ref_points:
        n_circles = 0
        line = draw_bresenham_line(x_center, y_center, ref_point[0], ref_point[1])

        for line_point in line:
            for circle in circles:
                if line_point in circle:
                    if grayscale[line_point] >= threshold:
                        n_circles += 1

        if len(circles) == n_circles:
            line = [pair for pair in line if pair[0] < img.shape[1] and pair[1] < img.shape[0]]
            ray_lines.append(line)


    # Check
    point_to_degree = 360 / len(ref_points)
    degree_to_point = 1 / point_to_degree

    point_sequence = sort_circle_points(ref_points)

    statistics = {}
    for angle in range(0, 365, alpha // 2):
        piece = point_sequence[round(angle * degree_to_point): round(angle * degree_to_point) + round(alpha * degree_to_point)]
        n_rays = 0
        for line in ray_lines:
            for point in piece:
                if point in line:
                    n_rays += 1

        statistics[angle] = n_rays

    ray_angle = sort_dict(statistics, byValue=True, reversed=True)[0]
    valid_points = point_sequence[round(ray_angle[0] * degree_to_point): round(ray_angle[0] * degree_to_point) + round(alpha * degree_to_point)]

    xray = []
    for line in ray_lines:
        for point in valid_points:
            if point in line:
                xray.append(line)

    # Draw actions
    if visualise:
        for circle in circles:
            for pixel in circle:
                img[pixel] = [0, 255, 0] if grayscale[pixel] >= threshold else [0, 0, 255]

        for line in xray:
            for point in line:
                img[point] = [0, 165, 255]


    # Return result
    return img

def main():

    DATA_PATH = "./data/"
    all_examples = [fn for fn in glob.glob(DATA_PATH + "*.png") if fn.endswith("300x300.png")]
    num_examples = 20
    offset = np.random.randint(30) + 80
    i = 0
    for example in all_examples[offset: offset + num_examples]:
        i += 1
        image = cv.imread(example, cv.IMREAD_COLOR)
        image = ray_detector(image, visualise=True)
        show_img(image)

        #cv.imwrite("./xray_test/test_" + str(i + offset) + ".png", np.hstack((original, image)))
        print(i, "/", num_examples)

if __name__ == "__main__":
    main()
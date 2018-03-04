import cv2 as cv
import numpy as np


def radial_mark_irrelevant(img=np.zeros((1, 1))): #FIXME
    ret, thresh_trunc = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)
    return thresh_trunc


def show_img(image):
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.resizeWindow('image', 600, 600)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def narrow_gaps(img):

    # image = cv2.imread(file, cv2.IMREAD_GRAYSCALE) !

    # Mask
    mask = cv.imread('../resources/narrow_gaps_mask.png', cv.IMREAD_GRAYSCALE)
    mask = np.uint8(np.absolute(mask))

    # Detecting squares
    laplacian = cv.Laplacian(img, cv.CV_64F)
    sobel_xy = cv.Sobel(img, cv.CV_64F, 1, 1, ksize=1 + 2 * 0)

    edge_x = np.uint8(np.absolute(sobel_xy))
    edge_y = np.uint8(np.absolute(laplacian))
    result = cv.bitwise_or(edge_x, edge_y)
    result = cv.bitwise_and(result, mask)

    # Normalizing and closing detected squares
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    result = cv.morphologyEx(result, cv.MORPH_CLOSE, kernel)

    threshold = 3   # MAGIC NUMBER!
    result[np.where(result > threshold)] = 255
    result[np.where(result <= threshold )] = 0

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    grid = result = cv.morphologyEx(result, cv.MORPH_CLOSE, kernel, iterations=4)

    # Contours
    _, contours, hierarchy = cv.findContours(result, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # Draw contours
    result = cv.cvtColor(result, cv.COLOR_GRAY2BGR)
    cv.drawContours(result, contours, -1, (0, 0, 255), thickness=1)

    # Draw squares
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)

    # Removing grid (squares)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(grid, (x, y), (x + w, y + h), (255), thickness=cv.FILLED)

    offsets = {
        'top': 0,
        'bottom': grid.shape[0] - 1,
        'left': 0,
        'right': grid.shape[1] - 1
    }

    while np.sum(grid[:, offsets['left']]) == 0:
        offsets['left'] += 1

    while np.sum(grid[:, offsets['right']]) == 0:
            offsets['right'] -= 1

    while np.sum(grid[offsets['bottom']]) == 0:
        offsets['bottom'] -= 1

    while np.sum(grid[offsets['top']]) == 0:
        offsets['top'] += 1


    #grid[offsets['top'], : ] = 120
    #grid[offsets['bottom'], : ] = 120
    #grid[:, offsets['left']] = 120
    #grid[:, offsets['right']] = 120

    temp = img.copy()
    grid = grid[offsets['top']: offsets['bottom'], offsets['left'] : offsets['right']]
    temp = temp[offsets['top']: offsets['bottom'], offsets['left'] : offsets['right']]

    rows = []
    cols = []

    for i in range(grid.shape[0]):
        if np.sum(grid[i, :]) == 0:
            rows.append(i)

    for i in range(grid.shape[1]):
        if np.sum(grid[:, i]) == 0:
            cols.append(i)

    # Temporary solution !
    grid_color = temp[rows[0], 0]

    while len(rows) >= 2:
        selected_rows = (rows.pop(), rows.pop())
        temp = np.delete(temp, selected_rows, axis=0)
        offsets['top'] += 1
        offsets['bottom'] += 1

    while len(cols) >= 2:
        selected_cols = (cols.pop(), cols.pop())
        temp = np.delete(temp, selected_cols, axis=1)
        offsets['left'] += 1
        offsets['right'] += 1

    new_img = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    new_img[:] = grid_color

    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            new_img[i + offsets['top'], j + offsets['left']] = temp[i, j]

    return new_img


def test_narrow_gaps():
    DATA_PATH = "./data/"
    examples = ['2Ux_24_2mHID5i.300x300', 'T2664_PeakH_1.300x300', '2Ux_24.300x300', '6F_2.300x300', '7V_2.300x300',
                '3507_P_001.300x300', '8893_2_E2.300x300', '9766_E1_2.300x300', '9815_1.300x300']
    for example in examples:
        file = DATA_PATH + example + ".png"
        image = cv.imread(file, cv.IMREAD_GRAYSCALE)
        result = narrow_gaps(image)
        show_img(image)
        show_img(result)

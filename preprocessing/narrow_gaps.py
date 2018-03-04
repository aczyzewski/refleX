import cv2
import numpy as np

DATA_PATH = "./data/"

def show_img(image):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def narrow_gaps(img):

    #image = cv2.imread(file, cv2.IMREAD_GRAYSCALE) !

    # Mask
    mask = cv2.imread('./preprocessing/narrow_gaps_mask.png', cv2.IMREAD_GRAYSCALE)
    mask = np.uint8(np.absolute(mask))

    # Detecting squares
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=1 + 2 * 0)

    edgeX = np.uint8(np.absolute(sobelxy))
    edgeY = np.uint8(np.absolute(laplacian))
    result = cv2.bitwise_or(edgeX, edgeY)
    result = cv2.bitwise_and(result, mask)

    # Normalizing and closing detected squares
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    threshold = 3   # MAGIC NUMBER!
    result[np.where(result > threshold)] = 255
    result[np.where(result <= threshold )] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grid = result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=4)

    # Contours
    _, contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Draw contours
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, contours, -1, (0, 0, 255), thickness=1)

    # Draw squares
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)

    # Removing grid (squares)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(grid, (x, y), (x + w, y + h), (255), thickness=cv2.FILLED)


    offsets = {
        'top' : 0,
        'bottom' : grid.shape[0] - 1,
        'left' : 0,
        'right' : grid.shape[1] - 1
    }

    while(np.sum(grid[:, offsets['left']]) == 0):
        offsets['left'] += 1

    while (np.sum(grid[:, offsets['right']]) == 0):
            offsets['right'] -= 1

    while(np.sum(grid[offsets['bottom']]) == 0):
        offsets['bottom'] -= 1

    while(np.sum(grid[offsets['top']]) == 0):
        offsets['top'] += 1


    #grid[offsets['top'], : ] = 120
    #grid[offsets['bottom'], : ] = 120
    #grid[:, offsets['left']] = 120
    #grid[:, offsets['right']] = 120

    temp = img.copy()
    grid = grid[offsets['top'] : offsets['bottom'], offsets['left'] : offsets['right']]
    temp = temp[offsets['top'] : offsets['bottom'], offsets['left'] : offsets['right']]

    rows = []
    cols = []

    for i in range(grid.shape[0]):
        if (np.sum(grid[i, :]) == 0):
            rows.append(i)

    for i in range(grid.shape[1]):
        if (np.sum(grid[:, i]) == 0):
            cols.append(i)

    # Temporary solution !
    grid_color = temp[rows[0], 0]

    while(len(rows) >= 2):
        selected_rows = (rows.pop(), rows.pop())
        temp = np.delete(temp, selected_rows, axis=0)
        offsets['top'] += 1
        offsets['bottom'] += 1


    while(len(cols) >= 2):
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

if __name__ == '__main__':

    examples = ['2Ux_24_2mHID5i.300x300', 'T2664_PeakH_1.300x300', '2Ux_24.300x300', '6F_2.300x300', '7V_2.300x300',
                '3507_P_001.300x300', '8893_2_E2.300x300', '9766_E1_2.300x300', '9815_1.300x300']

    for example in examples:

        file = DATA_PATH + example + ".png"

        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        result = narrow_gaps(image)
        show_img(image)
        show_img(result)

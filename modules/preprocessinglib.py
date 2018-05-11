# -*- coding: utf-8 -*-

import glob
import ntpath

import numpy as np
import cv2 as cv
import modules.util as util

from math import cos, sin, log
from sklearn.metrics.pairwise import cosine_distances


# TODO: Documentation
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


'''Ta funkcja w teorii ma błąd, ale w praktyce ładnie działa, więc byc może będziemy chcieli do niej wrócić'''
def logarithmize_old(img):
    l_img = np.zeros_like(img)
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            if pixel == 0:
                l_img[i, j] = 255
            else:
                l_img[i, j] = round(255*-log(pixel/255))
    return l_img


'''Maps each pixel of the input image to the negative of its logarithm, 
Args: img - grayscale image
Returns - grayscale image of the same shape as input img'''
def logarithmize(img):
    l_img = np.zeros_like(img)
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            if pixel:
                l_img[i, j] = round(255 * log(pixel, 255))
    return l_img


def t_logarithmize(img_filenames):

    for fn in img_filenames:
        print(fn)
        img = cv.imread(fn, cv.IMREAD_GRAYSCALE)
        util.show_img(np.hstack((img, logarithmize(img))))

    img = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        img[i, :] = i
    util.show_img(np.hstack((img, logarithmize(img))))


def extend_bresenham_line(image, line_segment, multipler=2, border=25):
    """ Rozszerza podany odcinek złożony z puntów o podany współczynnik mnożący.

    Funkcja przymuje listę punktów, które tworzą odcinek. Wyznacza następnie
    nowe punkty, które będą stanowiły końce nowego odcinka (zawierającego w sobie
    odcinek początkowy). Punkty (linia) pomiędzy nowym końcami wyznaczana jest
    za pomocą algorytmu Bresenhama.

    Funckcja docelowo została napisane w celu łatwego rozszerzania zbioru kandydatów
    na środek obrazu, stąd dodatkowe parametry takie jak `image` oraz `border`, które
    nakładały ograniczenia na odcinek, tak, aby spełniał wymagania projektu.

    Odcinek nie jest powiekszany równomiernie, a jedynie do do pierwszego punktu podanego
    odcinka dobudowywane są jego wielokrotnosci. Dla przykładu: `multipler=2` spowoduje
    dodanie 4 odcinkow (po 2 z kazdej strony) o dlugosciach odcinka poczatkowego, gdzie
    pierwszy punkt podanego początkowego odcinka będzie leżał na środku nowego odcinka.

    Args:
        image (numpy array):    Obraz, na który nakładany będzie nowy odcinek.
        line_segment (list):    Lista punktów tworzących odcinek.
        multipler (int):        Wyznacza ilukrotnie ma zostać powiększony podany odcinek
        border (int):           Określa najmniejszą dozwoloną odległość między końcem odcinka,
                                a krawędzią obrazu.

    Returns:
        (list):                 Lista punktów tworzących nowy odcinek

    """

    extend = [line_segment[0][0] - line_segment[-1][0], line_segment[0][1] - line_segment[-1][1]]
    reference_point = line_segment[0]

    first_end = (np.array(reference_point) + (multipler * np.array(extend))).tolist()
    second_end = (np.array(reference_point) - (multipler * np.array(extend))).tolist()

    extended_can_cord_set = util.bresenham_line_points(*first_end, *second_end)
    validated_extended_can_cord_set = [point for point in extended_can_cord_set if border < point[0] < image.shape[0] - border and border < point[1] < image.shape[0] - border]
    return validated_extended_can_cord_set


def find_ray_angle(img, center, num_samples=1024):
    """ Na podstawie podanego obrazu wyznacza kąt początkowy i końcowy padającego promienia.

    Na podstawie podanego środka zdjęcia, funkcja analizuje wszystkie wektory pixeli,
    które z niego wychodzą pod każdym możliwym kątem (domyślnie przypada 512 promieni
    na 360 stopni), a następnie wyznacza z te wektory, których średnia jasność 
    przekracza 4 odchylenia standardowe. Zwracane są kąty skrajnych wyznaczonych wektorów.

    Args:
        img (numpy array):      Obraz - w skali szarości.
        center (list):          Środek badanego obrazu w formacie (X, Y)
        num_samples:            Liczba rozpatrywanych wektorów-promieni.

    Returns:
        (int, int):             Wyznaczony kąt początkowy i końcowy. Między kątęm początkowym i końcowym (zgodnie ze
                                wskazówkami zegara) znajduje się promień.
                                Wartość `0` wskazuje na kierunek północny.
                                Kąt rośnie zgodnie ze wskazówkami zegara.

        None:                   Jeżeli nie istnieje wektor pixeli spełniajacy warunku
                                z czterema odchyleniami stadardowymi.

    """

    radii_coordinates = get_radii_coordinates(img, center, num_samples=num_samples, offset=0)
    radii_pixels = [[img[rc] for rc in radius_coords] for radius_coords in radii_coordinates]
    v_means = np.array([np.mean(radius) for radius in radii_pixels])

    mean, std = np.mean(v_means), np.std(v_means)
    v_means -= mean

    ray_angles = []
    for i in range(num_samples):
        if v_means[i] > 4 * std:
            ray_angles.append(360 * i / num_samples)

    if len(ray_angles) == 0:
        return None

    if ray_angles[-1] - ray_angles[0] > 360 - 3 * 360 / num_samples:
        return min(filter(lambda a: a > 180, ray_angles)), max(filter(lambda a: a <= 180, ray_angles))

    return ray_angles[0], ray_angles[-1]


def t_find_ray_angle(filenames, centers):
    from modules.util import show_img
    for fn, center in zip(filenames, centers):
        img = cv.imread(fn, cv.IMREAD_GRAYSCALE)
        fra = find_ray_angle(img, center)
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        cv.circle(img, center, radius=3, color=(0,0,255), thickness=-1)
        if fra:
            sa, ea = fra
            print("Angles", sa, ea)
            p1 = get_radii_coordinates(img, center, 1, sa)[0][-1][::-1]
            p2 = get_radii_coordinates(img, center, 1, ea)[0][-1][::-1]
            print('Points:', p1, p2)
            cv.line(img, center, p1, color=(255, 0, 0))
            cv.line(img, center, p2, color=(0, 255, 0))
        else:
            print("No ray")
        show_img(img)


# TODO: Uwzględnić `vector_variability(radii)`
def calculate_center(img, padding=10):
    """ Wyznacza środek obrazu dyfrakcyjnego.

    Funcja na podstawie puntów wyznaczonych za pomocą środka masy podanego zdjęcia
    oraz jego negatywu wyznacza odcinek, którego każdy punkt jest może być
    potencjalnym środkiem obrazu. Następnie dla każdego z tych puntków generuje się
    i analizuje wychodzące wektory pod każdym kątem o możliwe jak nawiększej długości.
    Punkt, który zebrał najwięcej "puntów" jest prawdopodobnie środkiem obrazu.


    Args:
        img (numpy array):          Obraz - w skali szarości.
        padding (int):              Określa procentowo ile najbliższych pikseli każdego wektora
                                    jest ignorowanych

    Returns:

        candidate_coordinate_set (list):    Zbiór kandydatów na środek
        candidate1 (tuple):                 Srodek masy obrazu
        candidate2  (tuple):                Srodek masy negacji obrazu
        best_candidate (tuple):             Rzeczywisty środek obrazu

    """

    l_img = logarithmize(img)
    candidate1 = util.center_of_mass(l_img)[::-1]
    candidate2 = util.center_of_mass(util.negative(l_img))[::-1]
    #logger.debug("Candidates are: " + str(candidate1) + " and " + str(candidate2))
    candidate_coordinate_set = extend_bresenham_line(img, util.bresenham_line_points(*candidate1, *candidate2), border=50)
    #logger.debug("Number of candidates: " + str(len(candidate_coordinate_set)))

    best_candidate = None
    min_variability = None
    values = []
    cords = []

    for candidate_coords in candidate_coordinate_set:

        radii_coordinates = get_radii_coordinates(img, candidate_coords, num_samples=256, offset=0)
        radii_pixels = [[img[xy] for xy in radius_coords]
                 for radius_coords in radii_coordinates]

        #current_variability = vector_variability(radii)
        x, y = candidate_coords

        # TODO: Minimalizacja wariancji
        border_size = int(img.shape[0] / 512)
        fileds = img[x - border_size : x + border_size, y - border_size : y + border_size].flatten()
        mean_center_pixel_border = sum(fileds) / len(fileds)
        padding_value = round(padding * len(radii_pixels[0]) / 100)
        current_variability = max([(sum(radius[padding_value:]) / len(radius[padding_value:])) for radius in radii_pixels]) * (abs(90 - (mean_center_pixel_border)) / 255)
        # TODO: Koniec -> Minimalizacja wariancji

        values.append(current_variability)
        cords.append(candidate_coords)

        if best_candidate is None or current_variability > min_variability:
            min_variability = current_variability
            best_candidate = candidate_coords

    return candidate_coordinate_set, candidate1, candidate2, best_candidate #FIXME remove redundant returns


'''Returns vector of radii, starting with north and going clockwise.
Args: center(x,y)
Returns numpy-style (y,x) format'''
def get_radii_coordinates(img, center, num_samples=20, offset=0):

    x, y = center
    if not 0 < x < img.shape[1] or not 0 < y < img.shape[0]:
        return []

    angles = np.linspace(0 + offset, 360 + offset, num_samples + 1)
    angles = angles + 180
    radius = np.min([img.shape[1] - x - 1, x, img.shape[0] - y - 1, y])

    vectors = []
    for angle in angles:
        vector = []
        for dist in np.linspace(1, radius):
            floating_point = (x + (dist * cos(np.radians(angle))),  y - (dist * sin(np.radians(angle))))
            vector.append(util.closest_pixel(floating_point))
        vectors.append(vector)

    return vectors


def t_get_radii_coordinates(result=np.zeros((500,500)), center=(250, 250)):
    ray_coords = get_radii_coordinates(result, center, 1, 10)[0]
    cv.line(result, center[::-1], ray_coords[-1][::-1], (255, 255, 255))
    ray_coords = get_radii_coordinates(result, center, 1, 30)[0]
    cv.line(result, center[::-1], ray_coords[-1][::-1], (0, 0, 255))
    ray_coords = get_radii_coordinates(result, center, 1, 45)[0]
    cv.line(result, center[::-1], ray_coords[-1][::-1], (0, 255, 0))
    ray_coords = get_radii_coordinates(result, center, 1, 120)[0]
    cv.line(result, center[::-1], ray_coords[-1][::-1], (255, 0, 0))
    ray_coords = get_radii_coordinates(result, center, 1, 150)[0]
    cv.line(result, center[::-1], ray_coords[-1][::-1], (255, 0, 255))


# TODO: Documentation
#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
def vector_variability(vectors=[]):
    v = np.asmatrix(vectors)
    pairwise_cosine_similarity_matrix = cosine_distances(v)
    flattened = np.array([], dtype=np.float64)
    for i, row in enumerate(pairwise_cosine_similarity_matrix):
        flattened = np.concatenate((flattened, row[i+1:]), axis=0)

    return np.median(flattened)


# TODO: Documentation
def center_visualization(img, padding=10, additional_grid=False):

    # Obie wersje tego samego obrazu
    color_img = img.copy()
    color_img = cv.cvtColor(color_img, cv.COLOR_GRAY2BGR)

    # Wyznaczanie środka
    candidate_coordinate_set, candidate1, candidate2, center = calculate_center(img, padding)
    vectors = get_radii_coordinates(img, center, num_samples=512, offset=0)

    # Obliczanie ile % punktow poczatkowym punktow promieni odrzucamy
    # Wartosc ta musi byc zgodna z parametrem funkcji wyzej, gdyz ona
    # rowniez wycina tyle samo i na tej podstawie wyznacza srodek
    padding_value = round(padding * len(vectors[0]) / 100)

    # Rysowywanie wektorow wychodzacych z punktu
    for vector in vectors:
        for pixel in vector[padding_value:]:
            color_img[pixel] = [0, 155, 0]

    # Rysowanie zbioru kondydatow (linia prosta)
    for candidate in candidate_coordinate_set:
        color_img[candidate] = [0, 255, 0]

    # Wyrysowywanie srodkow
    cv.circle(color_img, center[::-1], 3, (255, 0, 0), thickness=-1)      # Wyznaczony srodek obrazu
    cv.circle(color_img, candidate1, 2, (255, 255, 255), thickness=-1)    # Wyznaczony srodek masy obrazu bez negacji
    cv.circle(color_img, candidate2, 2, (0, 0, 0), thickness=-1)          # Wyznaczony srodek masy obrazu po negacji

    # Rysowanie dodatkowego gridu
    # Kolor bialy: wyznaczony srodek obrazu
    # Kolor czarny: srodek obrazu (shape / 2)
    # Kolejnosc rysowania: bialy, czary
    if additional_grid:
        color_img = util.draw_grid(color_img, center)

    # Rysowanie lini pokrywającej się z padajacym promieniem o ile został wykryty
    start_angle, end_angle = find_ray_angle(img, center) #TODO check - zmiana sposobu wyznaczania zmiennej angle po zmianie wartosci zwracanych przez funkcje find_ray_angle
    angle = start_angle + (end_angle - start_angle)/2 if start_angle < end_angle \
        else (start_angle + (360 + end_angle - start_angle)/2) % 360
    if angle:
        ray_coords = get_radii_coordinates(img, center, 1, angle)[0]
        cv.line(color_img, center[::-1], ray_coords[-1][::-1], (0, 0, 255))

    return color_img


#FIXME remove
def test(testcasesdir, rawsourcedir):

    testcases = [ntpath.split(fn)[1] for fn in glob.glob(testcasesdir + "*.png")]
    allfiles = [fn for fn in glob.glob(rawsourcedir + "*.png")]
    all = True

    for im_name in allfiles:
        if ntpath.split(im_name)[1] in testcases or all:

            original_image = cv.imread(im_name, cv.IMREAD_GRAYSCALE)
            result = center_visualization(original_image, padding=10, additional_grid=True)

            filename = ntpath.split(im_name)[1]
            print(filename)

            # cv.imwrite('./test_ray/' + filename, result)
            # --- OR ---
            from modules.util import show_img
            show_img(result)


def main(dirname="./data/"):

    file_names = [fn for fn in glob.glob(dirname + "*.png")]
    #labeled_image_names = pd.read_csv("reflex.csv").iloc[:, 0].str.slice(7, -4).values

    for im_name in file_names[::-1]:
        print(im_name)
        img = cv.imread(im_name, cv.IMREAD_GRAYSCALE)
        #logger.debug("Image " + im_name + " read successfully")
        candidate_coordinate_set, candidate1, candidate2, center = calculate_center(img)
        print(center)
        data = {"image_name": im_name[len(dirname):], "x": center[1], "y": center[0]}
        util.write_to_csv("centers_ac.csv", data)


if __name__ == "__main__":
    #test('/Volumes/DATA/reflex_data/best/', '/Volumes/DATA/reflex_data/reflex_img_512_inter_nearest/')
    t_filenames = ["/Volumes/Alice/reflex-data/data_512/zza1-8_1_001.512x512.png",
                    "/Volumes/Alice/reflex-data/data_512/YUP_6_1_001.512x512.png",
                    "/Volumes/Alice/reflex-data/data_512/x1-high.0001.512x512.png",
                    "/Volumes/Alice/reflex-data/data_512/ProlWT_Mut0_HR_1_00001.512x512.png",
                    "/Volumes/Alice/reflex-data/data_512/bjp_plate2-b3_LR_8_001.512x512.png"]
    t_centers = [(254, 253), (255, 257), (266, 248), (261, 251), (261, 254)]
    #t_logarithmize(t_filenames)
    t_find_ray_angle(t_filenames, t_centers)
    pass
import glob 
import pandas as pd
import cv2 as cv

'''
Constructs a DataFrame from all vector files in the given directory.
'''
def vector_folder_to_df(directory, limit=None):

    files = list(os.listdir(directory + '/' + list(os.listdir(directory))[0]))
    all_dfs = []

    for idx, image in enumerate(files[:limit]): 
        all_dfs.append(create_joint_vector(image, directory, all_statistics=True, sort=True, vector_length=240, remove_suffix=True, image_as_index=True))

    return pd.concat(all_dfs)


def extract_filename(path):
    path, filename = os.path.split(path)
    return '.'.join(filename.split('.')[:-1])


def construct_joint_csv(output_filepath): # FIXME not used
    df1 = pd.read_csv('/home/reflex/reflex/data/results_constant_vector_length/vectors/vectors.csv')
    df1.set_index('img', inplace=True, drop=True)

    df2 = pd.read_csv('/home/reflex/reflex/reflex.csv')
    df2['Image'] = df2['Image'].apply(lambda x: extract_filename(x))
    df2.set_index('Image', inplace=True, drop=True)

    joint = pd.concat([df1, df2], axis=1, join='inner')
    joint.to_csv(output_filepath)
    return joint


def create_joint_vector(image, directory, statistics=None, all_statistics=False, sort=False, 
                    sort_function=lambda x: x, vector_length=None, 
                    fill_value=None, image_as_index=True, predefined_file_list=None, 
                    remove_suffix=False):

    """
    Tworzy dataframe z wybranego zdjecia w postaci <nazwa_zdjecia> <wektor_zlaczonych_statystyk>

    Params:
    --------------
    image                 - nazwa pliku ze zdjeciem (z suffixem (.SSSxSSS.png))
    directory             - katalog z katalogami ze wszystkimi statystykami
    statistics            - lista statystyk branych pod uwage
    all_statistics        - jezeli True to poprzedni parametr jest ignorowany i pod uwage bierzemy
                            wszystkie statystyki we wskazanym folderze
    sort                  - jezeli True to nazwy statystyk sortowane sa wedlug podanej funkcji
                            sortujacej (domyslnie leksykograficznie)
    sort_function         - funkcja (key) sortujaca statystyki (domyslnie leksykograficznie)
    vector_length         - parametr okreslajacy pozadana dlugosc wektora. W przypadku 
                            nadmiaru jest przycinany, w przeciwnym razie jest wypelniniany
                            kolejnym paremetrem. Niezdefiniowany (None) nie modyfikuje wektora.
    fill_value            - wartosc, ktora wypelniany bedzie wektor, jezeli będzie za krótki
                            w przypadku zdefiniowania dlugości
    image_as_index        - ustawie nazwe pliku jako index DataFrame'u
    remove_suffix         - usuwa suffix (.SSSxSSS.png) z nazwy pliku


    Returns:
    --------------
    None                  - w przpadku bledu (brak zdef. statystyk, zly katalog, zla nazwa pliku)
    Dataframe             - kiedy wszystko poszlo zgodnie z zalozeniami

    """

    statistic_names = list(os.listdir(directory)) if all_statistics else statistics 

    if statistic_names and sort:
        statistic_names.sort(key=sort_function)

    values = []

    # Laczenie wektora
    for stat_name in statistic_names:

        current_vector = cv.imread(os.path.join(directory, stat_name, image), cv.IMREAD_GRAYSCALE).flatten().tolist()

        # Dostosowywanie dlugosci wektora
        if vector_length:
            current_vector = current_vector[:vector_length]
            current_vector.extend([fill_value] * max(0, vector_length - len(current_vector)))

        values.append(current_vector)

    if remove_suffix:
        image = image[:-12]

    data = []
    column_names = ['img']
    for idx, stat_vec in enumerate(values):
        column_names += [f'{statistic_names[idx]}_{i}' for i in range(len(stat_vec))]
        data += stat_vec

    df = pd.DataFrame([[image, *data]], columns=column_names)

    if image_as_index:
        df.set_index('img', inplace=True)

    return df
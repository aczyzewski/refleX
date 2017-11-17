import glob
import logging
import subprocess
import os
import signal
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import misc
from sklearn import cluster
from collections import OrderedDict


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
SEED = 23
LABELING_FOLDER = os.path.join(os.path.dirname(__file__))
LABELING_FILE = os.path.join(LABELING_FOLDER, "reflex.csv")
LABELS = OrderedDict(sorted({
    "1": "Loop scattering",
    "2": "Background ring",
    "3": "Strong background",
    "4": "Diffuse scattering",
    "5": "Artifact",
    "6": "Ice ring",
    "7": "Non-uniform detector",
}.items()))


def write_to_csv(image_path, labels, file_path=LABELING_FILE):
    save_to_folder = os.path.dirname(file_path)

    if not os.path.exists(save_to_folder):
        os.mkdir(save_to_folder)

    if os.path.isfile(file_path):
        write_header = False
        mode = "a"
    else:
        write_header = True
        mode = "w"

    with open(file_path, mode) as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC, lineterminator='\n')

        if write_header:
            header = ["Image"]
            header.extend(LABELS.values())
            writer.writerow(header)

        row = [image_path]
        label_flags = [0] * LABELS.__len__()
        for label in labels:
            label_flags[int(label)-1] = 1
        row.extend(label_flags)
        writer.writerow(row)


def get_data_from_png(file_list):
    logging.info("Reading images from PNG files")

    images = []
    image_paths = []
    for image_path in file_list:
        image_paths.append(image_path)
        images.append(misc.imread(image_path, flatten=True))

    return np.stack(images), image_paths


def normalize_data(dataset):
    logging.info("Normalizing images")

    dataset_size, img_x, img_y = dataset.shape
    depth = 1
    dataset = dataset.reshape(dataset_size, img_x, img_y, depth)
    dataset = dataset.astype('float32')
    dataset /= 255

    return dataset, dataset_size, img_x, img_y, depth


def cluster_images(paths, X, dataset_size, img_x, img_y, depth, k, n_jobs=-1):
    logging.info("Clustering %d images into %d groups", dataset_size, k)
    X = np.reshape(X, (dataset_size, img_x*img_y*depth))
    k_means = cluster.KMeans(n_clusters=k, max_iter=50, n_jobs=n_jobs, random_state=SEED)
    k_means.fit(X)

    for i in range(k_means.n_clusters):
        with open("cluster_" + str(i) + ".txt", "w") as cluster_file:
            for label_idx in range(dataset_size):
                if k_means.labels_[label_idx] == i:
                    cluster_file.write("%s\n" % paths[label_idx])

    return k_means


def analyze_clusters(k_means, img_x, img_y):
    unique, counts = np.unique(k_means.labels_, return_counts=True)
    label_dict = dict(zip(unique, counts))

    for i in range(k_means.n_clusters):
        print "Cluster %d contains %d images" % (i, label_dict[i])
        cluster_center = k_means.cluster_centers_[i]
        cluster_center_img = cluster_center.reshape(img_x, img_y)
        cluster_center_img *= 255
        cluster_center_img = cluster_center_img.astype('uint8')
        plt.imshow(cluster_center_img, cmap=plt.cm.gray)
        plt.show()


def label_images(files):
    msg = "Enter labels(" + ", ".join([k + ":" + v for k, v in LABELS.iteritems()]) + "): "
    if os.path.isfile(LABELING_FILE):
        prev_processed_files = set(pd.read_csv(LABELING_FILE)["Image"])
    else:
        prev_processed_files = None

    for image_path in files:
        if prev_processed_files is not None and image_path in prev_processed_files:
            continue

        print
        print image_path
        pro = subprocess.Popen("xdg-open " + image_path, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
        labels = list(raw_input(msg))
        write_to_csv(image_path, labels)
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)


if __name__ == "__main__":
    files = [fn for fn in glob.glob("./data/*.png") if not (fn.endswith("100x100.png") or fn.endswith("300x300.png"))]

    # dataset, paths = get_data_from_png(files)
    # dataset, dataset_size, img_x, img_y, depth = normalize_data(dataset)
    # k_means = cluster_images(paths, dataset, dataset_size, img_x, img_y, depth, 15)
    # analyze_clusters(k_means, img_x, img_y)

    label_images(files)



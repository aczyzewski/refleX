import glob
import logging
import matplotlib.pyplot as plt
import numpy as np

from scipy import misc
from sklearn import cluster


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
SEED = 23


def get_data_from_png(file_mask):
    logging.info("Reading images from PNG files")

    images = []
    image_paths = []
    for image_path in glob.glob(file_mask):
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


def cluster_images(X, dataset_size, img_x, img_y, depth, k, n_jobs=-1):
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


def analyze_clusters(k_means):
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


if __name__ == "__main__":
    k = 15

    dataset, paths = get_data_from_png("./data/*100x100.png")
    dataset, dataset_size, img_x, img_y, depth = normalize_data(dataset)
    k_means = cluster_images(dataset, dataset_size, img_x, img_y, depth, k)
    analyze_clusters(k_means)

import glob
import logging
import subprocess
import os
import signal
import csv
import pandas as pd
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
    label_images(files)

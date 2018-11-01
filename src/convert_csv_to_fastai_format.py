import argparse
import ntpath
import sys
import os
import numpy as np
from util import train_test_split

def split_dataset(rows, fraction):

    x_values, y_values = [], []

    for row in rows:
        x_values.append(row[0])
        y_values.append(row[1:])
  
    X_train, X_test, y_train, y_test = train_test_split(x_values, y_values, test_size=fraction, stratify=y_values)

    train_size = len(X_train)
    test_size = len(X_test)

    X_train += X_test
    y_train += y_train

    output_data = []
    for img_name, anomalies in list(zip(X_train, y_train)):
        final_row = [img_name] + anomalies
        output_data.append(final_row)
    
    return [train_size, test_size, output_data]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    anomalies = ["loop_scattering", "background_ring", "strong_background", "diffuse_scattering", "artifact", "ice_ring", "non_uniform_detector"]

    parser.add_argument('-f', '--input_file', required=True, default='reflex.csv', help="Input file (categorical variable as dummy/indicator variables)")
    parser.add_argument('-o', '--output_file', default='fastai_reflex.csv', help="Output file name")
    parser.add_argument('-s', '--split_size', default=None, help='Split matrices into random (stratified) train and test subsets. This argument detemines fraction of test images.')
    parser.add_argument('-r', '--reduce_to', default=None, help='Intersection between files in CSV and files in selected directory')
    args = parser.parse_args()

    csv_data = open(args.input_file, 'r').read().split('\n')
    csv_data_splitted = [line.split(',') for line in csv_data if len(line)]
    csv_size = len(csv_data)
    print("[INFO] Read %d files" % csv_size)

    if not args.reduce_to is None and os.path.isdir(args.reduce_to):
        existing_files = [ntpath.split(file)[1][:-4] for file in os.listdir(args.reduce_to)]
        csv_data_splitted = [file for file in csv_data_splitted if file[0] in existing_files]
        print("[INFO] Filtered out %d files. (%s)" % (csv_size - len(csv_data_splitted), len(csv_data_splitted)))
    elif not args.reduce_to is None and not os.path.isdir(args.reduce_to):
            print("[WARINING] Argument -r (--reduced_to) is incorrect! Omitting...")

    if args.split_size:
        try:
            test_subset_size = float(args.split_size) if float(args.split_size) < 1.0 else float(args.split_size) / 100 
            if test_subset_size > 1.0 or test_subset_size < 0:
                raise Exception
            
            train_size, test_size, csv_data_splitted = split_dataset(csv_data_splitted, test_subset_size) 
            print('[INFO] Train subset size: %s [0 - %s] | Test subset size: %s [%s - %s]' % 
                (train_size, train_size - 1, test_size, train_size, train_size + test_size - 1))
        except Exception as e:
            print("[WARINING] Argument -s (--split_size) is incorrect! Omitting...")

    output_file = open(args.output_file, 'w')

    for row in csv_data_splitted:
        preditions = ' '.join([anomaly for idx, anomaly in enumerate(anomalies) if int(row[idx + 1])])
        output_file.write(",".join([row[0], preditions]) + "\n")

    output_file.close()
    print("[INFO] Done.")




    


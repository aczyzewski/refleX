import os
import sys
import warnings
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import datetime
import traceback
import argparse

from custom_metrics import f2, average_precision, average_recall, hamming_score

sys.path.insert(0, '/home/reflex/refleX/lib/fastai')
from fastai.conv_learner import *

def save_lr_chart(lrs, losses, filename, n_skip=10, n_skip_end=5):
    plt.ylabel("validation loss")
    plt.xlabel("learning rate (log scale)")
    plt.plot(lrs[n_skip:-(n_skip_end+1)], losses[n_skip:-(n_skip_end+1)])
    plt.xscale('log')
    plt.savefig(filename)
    plt.clf()

def get_data (model, PATHS, img_size, custom_transformations, batch_size, val_idxs):
    tfms = tfms_from_model(model, img_size, aug_tfms=custom_transformations, max_zoom=1.1)
    return ImageClassifierData.from_csv(PATHS['DATA_PATH'], PATHS['DATA_FOLDER'], PATHS['CSV_PATH'], tfms=tfms,
                    suffix='.png', val_idxs=val_idxs, bs=batch_size, num_workers=2)


def grid_search(PATHS, CONFIG, experiment_name, PARAMS, output_csv='results.csv'):   

    val_idxs = PARAMS['val_idxs']

    # TODO: Czy LR wyznaczamy przed na samym początku czy przed kazdym uczeniem (tj. przed zmiana rozdzielczosci?)

    # Logger
    logger = logging.getLogger(f'{experiment_name}')
    logging.basicConfig(filename=f'{PATHS["EXP_PATH"]}/{experiment_name}/{experiment_name}.log', format='%(asctime)s - %(levelname)s - %(message)s',\
        filemode='w', datefmt='%d-%m-%Y %I:%M:%S %p', level=logging.INFO)   

    # Dodatowe parametry
    dropout = PARAMS['dropout']
    earlystoping = PARAMS['earlystopping']
    log_step = PARAMS['log_step']
    custom_aug_tfms = PARAMS['transformations']
    gpu = PARAMS['gpu_id']
    torch.cuda.set_device(gpu)

    # 1. INFO
    logging.info(f" [ ------ EXPERIMENT: {experiment_name.upper()} ------ ] \n")

    # Log
    logging.info(' --- PARAMETERS: --- ')
    logging.info(f"Predefined metrics: [val_loss], {', '.join([metric.__name__ for metric in CONFIG['predefined_metrics']])}")
    logging.info(f"Predefined architecutres: {', '.join([architecture[0] for architecture in CONFIG['architectures'].items()])}")
    logging.info(f"Predefined batch sizes: {', '.join([str(batch_size) for batch_size in CONFIG['batch_sizes']])}")
    logging.info(f"Predefined image sizes: {', '.join([str(img_size) for img_size in CONFIG['image_sizes']])}")
    logging.info(f"Dropout: {dropout}")
    logging.info(f"Earlystopping after: {earlystoping} epochs.")
    logging.info(f"Log every {log_step} epochs.")
    logging.info(f'GPU: {torch.cuda.current_device()} | Available: {torch.cuda.device_count()} \n')


    for model_name, model in CONFIG['architectures'].items():

        logging.info(f"    ----- *** {model_name.upper()} *** ----- \n")

        # Definiujemy obiekt learner'a
        data = get_data(model, PATHS, CONFIG['image_sizes'][0], custom_aug_tfms, CONFIG['batch_sizes'][0], val_idxs)
        learn = ConvLearner.pretrained(model, data, ps=dropout, metrics=CONFIG['predefined_metrics'])
        
        # Obliczanie LR - wybieranie najmniejszego loss i dzielenie przez 10
        learn.lr_find()
        save_lr_chart(learn.sched.lrs, learn.sched.losses, f'{PATHS["EXP_PATH"]}/{experiment_name}/{model_name}_lr_sched.png')

        lr = sorted(list(zip(learn.sched.lrs, learn.sched.losses)), key=lambda x: x[1])[0][0] / 10
        lrs = [lr/8, lr/4, lr/2, lr]

        # Log
        logging.info(f' Learning rate = {round(lr, 6)}')
        logging.info(f' Learning rates = {[round(lr, 6) for lr in lrs]} \n')

        for batch_size in CONFIG['batch_sizes']:

            for img_size in CONFIG['image_sizes']:

                try:
                    logging.info(f" --- Batch: {batch_size} | Img size: {img_size} --- \n ")

                    # Preparing data
                    best_model_name = f'{PATHS["EXP_PATH"]}/{experiment_name}/models/best_{model_name}_imgsize{img_size}_batch{batch_size}'
                    data = get_data(model, PATHS, img_size, custom_aug_tfms, batch_size, val_idxs)
                    learn.set_data(data)

                    # Freeze
                    logging.info(f" --- Type: FREEZE --- ")
                    learn.freeze()
                    learn.fit(lr, 12, cycle_len=1, cycle_mult=2, best_save_name=f'{best_model_name}_freeze', earlystopping=earlystoping, logger_name=experiment_name, log_step=log_step)
                    
                    # Print results
                    results = open(f'{best_model_name}_freeze_metrics.log', 'r').read()
                    logging.info(f"[ Saved model: best_{model_name}_imgsize{img_size}_batch{batch_size}_freeze.h5 ] ")
                    logging.info(f'[ Results: {results} ]\n')

                    # Unfreeze
                    logging.info(f" --- Type: UNFREEZE --- ")
                    learn.unfreeze()
                    learn.fit(lr, 12, cycle_len=1, cycle_mult=2, best_save_name=f'{best_model_name}_unfreeze', earlystopping=earlystoping, logger_name=experiment_name, log_step=log_step)
                    
                    # Print results
                    results = open(f'{best_model_name}_unfreeze_metrics.log', 'r').read()
                    logging.info(f"[ Saved model: best_{model_name}_imgsize{img_size}_batch{batch_size}_unfreeze.h5 ]")
                    logging.info(f'[ Results: {results} ]\n')

                    if img_size == CONFIG['image_sizes'][-1]:
                        logging.info(" ************************** ")
                        logging.info(f'COPYPASTE: {model_name} | IMS: {CONFIG["image_sizes"][0]} -> {img_size} | BS: {batch_size}')
                        logging.info(f'COPYPASTE: {results} ')
                        logging.info(" ************************** \n")

                except Exception as e:
                    logging.error(f'ERROR: A: {model_name} | IS: {img_size} | BS: {batch_size}')
                    logging.error(traceback.format_exc())

            if batch_size != CONFIG['batch_sizes'][-1]:
                logging.info(' --- INCREMENTING BATCH SIZE --- \n')

        logging.info(f"    ----- *** END: {model_name} *** ----- \n")

    print("[ --- DONE! --- ]")
                

def polar_configuration():
    paths = {
        'CSV_PATH': '/home/reflex/refleX/metadata/csv/augmented_sample_reflex.csv',
        'DATA_FOLDER' : 'augmented_polar512_sample'
        } # PARAMETER
    params = {'transformations' : None}
    config = {
        'image_sizes': [64, 128, 256, 512],
        'predefined_metrics': [f2, average_precision, average_recall, hamming_score],
        'batch_sizes': [8, 16, 32, 64], # PARAMETER
        'architectures' : {
            'resnet34' : resnet34 # PARAMETER
        }
    }

    return params, paths, config

def ac_resnet_configuration():
    paths = {
        'CSV_PATH': '/home/reflex/refleX/metadata/csv/fastai_validate_train.csv',
        'DATA_PATH': '/home/reflex/refleX/metadata/labeled/',
        'DATA_FOLDER': 'original512',
        'EXP_PATH': '/home/reflex/refleX/results/fastai_experiments'
    }

    params = {
        'val_idxs': range(200),
        'dropout': 0.5,
        'earlystopping': 12,
        'log_step': 1,
        'gpu_id': 1,
        'transformations' : [RandomLighting(0.1, 0.1), RandomDihedral()]
    }

    config = {
        'image_sizes': [64, 128, 256, 512],
        'predefined_metrics': [f2, average_precision, average_recall, hamming_score],
        'batch_sizes': [8, 16],
        'architectures' : {
            'resnet18' : resnet18,
            'resnet34' : resnet34
        }
    }

    return params, paths, config


def ac_polar_configuration():
    paths = {
        'CSV_PATH': '/home/reflex/refleX/metadata/csv/fastai_validate_train.csv',
        'DATA_PATH': '/home/reflex/refleX/metadata/labeled/',
        'DATA_FOLDER': 'polar512',
        'EXP_PATH': '/home/reflex/refleX/results/fastai_experiments'
    }

    params = {
        'dropout': 0.5,
        'earlystopping': 12,
        'log_step': 1,
        'gpu_id': 2,
        'transformations' : [RandomLighting(0.05, 0.05)]
    }

    config = {
        'image_sizes': [64, 128, 256, 512],
        'predefined_metrics': [f2, average_precision, average_recall, hamming_score],
        'batch_sizes': [8, 16, 32, 64],
        'architectures' : {
            'resnet18' : resnet18,
            'resnet34' : resnet34
        }
    }

    return params, paths, config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('working_title')
    parser.add_argument('-g', '--gpu_id', default=0, help="GPU ID")
    parser.add_argument('-cc', '--custom_config', default=None, help="Path to custom configuration file.")
    parser.add_argument('-cn', '--config_name', default='test', help="Name of hardcoded configuration")
    args = parser.parse_args()

    CUSTOM_PARAMS, CUSTOM_PATHS, CONFIG = globals()[args.config_name + "_configuration"]()

    #  --- DEFAULT PATHS ---
    PATHS = {
        'CSV_PATH': '/home/reflex/refleX/metadata/csv/fastai_train_test_split20.csv',
        'DATA_PATH': '/home/reflex/refleX/metadata/labeled/',
        'DATA_FOLDER': 'original512',
        'EXP_PATH': '/home/reflex/refleX/results/fastai_experiments'
    }

    # --- APPLY CUSTOM PATHS  ---
    for custom_path in CUSTOM_PATHS.keys():
        PATHS[custom_path] = CUSTOM_PATHS[custom_path]

    # --- DEFAULT PARAMS ---
    PARAMS = {
        'val_idxs': range(200),
        'dropout': 0.5,
        'earlystopping': 12,
        'log_step': 5,
        'gpu_id': args.gpu_id,
        'transformations' : [RandomLighting(0.1, 0.1), RandomDihedral()]
    }

    # --- APPLY CUSTOM PARAMETERS ---
    for custom_param in CUSTOM_PARAMS.keys():
        PARAMS[custom_param] = CUSTOM_PARAMS[custom_param]

    # Create dirs
    current_instance_dir = f'{PATHS["EXP_PATH"]}/{args.working_title}'
    models_dir = f'{PATHS["EXP_PATH"]}/{args.working_title}/models/'
    tmp_dir = f'{PATHS["EXP_PATH"]}/{args.working_title}/tmp/'

    os.makedirs(current_instance_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # Set validation set
    # convert_csv_to_fastai_format.py

    # Run grid search
    grid_search(PATHS, CONFIG, args.working_title, PARAMS)
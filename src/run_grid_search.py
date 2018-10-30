import os
import sys
import warnings
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import datetime
import traceback

from custom_metrics import f2, average_precision, average_recall, hamming_score

sys.path.insert(0,'/home/reflex/fastai/courses/dl1')
from fastai.conv_learner import *

def save_lr_chart(lrs, losses, filename, n_skip=10, n_skip_end=5):
    plt.ylabel("validation loss")
    plt.xlabel("learning rate (log scale)")
    plt.plot(lrs[n_skip:-(n_skip_end+1)], losses[n_skip:-(n_skip_end+1)])
    plt.xscale('log')
    plt.savefig(filename)

def get_data (model, data_path, csv_path, img_size, custom_transformations, batch_size, val_idxs, workers=2):
    tfms = tfms_from_model(model, img_size, aug_tfms=custom_transformations, max_zoom=1.1)
    return ImageClassifierData.from_csv(data_path, 'polar512', csv_path, tfms=tfms, # TODO hardcoded
                    suffix='.png', val_idxs=val_idxs, bs=batch_size, num_workers=workers)

def grid_search(data_path, csv_path, image_sizes, predefined_metrics, batch_sizes, architecures, val_idxs, experiment_name, **kwargs):   

    """
        Args:
        --------
            data_path           - sciezka do katalogu, gdzie przechowywane beda wszysktkie dane
                                np. zapisane modele, zapisane dane uczace, etc.
            csv_path            - sciezka do pliku CSV, gdzie zdefiniowane sa dane uczace
            image_sizes         - lista rozpatrywanych rozdzielczosci obrazow
            predefined_metrics  - zdefiniowane metryki
            batch_sizes         - lista rozpatrywanych rozmiarow batch'a
            architectures       - architektury sieci brane pod uwage
    """

    # TODO: Czy LR wyznaczamy przed na samym początku czy przed kazdym uczeniem (tj. przed zmiana rozdzielczosci?)

    # Logger
    logger = logging.getLogger(f'{experiment_name}')
    logging.basicConfig(filename=f'{data_path}/{experiment_name}/{experiment_name}.log', format='%(asctime)s - %(levelname)s - %(message)s',\
        filemode='w', datefmt='%d-%m-%Y %I:%M:%S %p', level=logging.INFO)   

    # Dodatowe parametry
    dropout = kwargs.get('dropout', 0.5)
    earlystoping = kwargs.get('earlystoping', 10)
    log_step = kwargs.get('log_step', 10)
    gpu = kwargs.get('gpu', 0)
    torch.cuda.set_device(gpu)
    custom_aug_tfms = None

    # 1. INFO
    logging.info(f" [ ------ EXPERIMENT: {experiment_name.upper()} ------ ] \n")

    # Log
    logging.info(' --- PARAMETERS: --- ')
    logging.info(f"Predefined metrics: [val_loss], {', '.join([metric.__name__ for metric in predefined_metrics])}")
    logging.info(f"Predefined architecutres: {', '.join([architecture[0] for architecture in architecures.items()])}")
    logging.info(f"Predefined batch sizes: {', '.join([str(batch_size) for batch_size in batch_sizes])}")
    logging.info(f"Predefined image sizes: {', '.join([str(img_size) for img_size in image_sizes])}")
    logging.info(f"Dropout: {dropout}")
    logging.info(f"Earlystopping after: {earlystoping} epochs.")
    logging.info(f"Log every {log_step} epochs.")
    logging.info(f'GPU: {torch.cuda.current_device()} | Available: {torch.cuda.device_count()} \n')


    for model_name, model in architecures.items():

        logging.info(f"    ----- *** {model_name.upper()} *** ----- \n")

        # Definiujemy obiekt learner'a
        data = get_data(model, data_path, csv_path, image_sizes[0], custom_aug_tfms, batch_sizes[0], val_idxs)
        learn = ConvLearner.pretrained(model, data, ps=dropout, metrics=predefined_metrics)
        
        # Obliczanie LR - wybieranie najmniejszego loss i dzielenie przez 10
        learn.lr_find()
        save_lr_chart(learn.sched.lrs, learn.sched.losses, f'{data_path}/{experiment_name}/{model_name}_lr_sched.png')

        lr = sorted(list(zip(learn.sched.lrs, learn.sched.losses)), key=lambda x: x[1])[0][0] / 10
        lrs = [lr/10, lr/5, lr/3, lr]

        # Log
        logging.info(f' Learning rate = {round(lr, 6)}')
        logging.info(f' Learning rates = {[round(lr, 6) for lr in lrs]} \n')

        for batch_size in batch_sizes:

            for img_size in image_sizes:

                try:
                    logging.info(f" --- Batch: {batch_size} | Img size: {img_size} --- \n ")

                    # Preparing data
                    best_model_name = f'{data_path}/{experiment_name}/models/best_{model_name}_imgsize{img_size}_batch{batch_size}'
                    data = get_data(model, data_path, csv_path, img_size, custom_aug_tfms, batch_size, val_idxs)
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
                    
                    if img_size == image_sizes[-1]:
                        logging.info(" ************************** ")
                        logging.info(f'COPYPASTE: {model_name} | IM: {img_size} | BS: {batch_size}')
                        logging.info(f'COPYPASTE: {results} ')
                        logging.info(" ************************** \n")
                    else:
                        logging.info(f'[ Results: {results} ]\n')

                except Exception as e:
                    logging.error(f'ERROR: A: {model_name} | IS: {img_size} | BS: {batch_size}')
                    logging.error(traceback.format_exc())

            if batch_size != batch_sizes[-1]:
                logging.info(' --- INCREMENTING BATCH SIZE --- \n')

        logging.info(f"    ----- *** END: {model_name} *** ----- \n")

    print("[ --- DONE! --- ]")
                

def test_configuration():
    image_sizes = [32, 64]
    metrics = [f2, average_precision, average_recall, hamming_score]
    batch_sizes = [32, 64]
    arch = {
            'resnet18' : resnet18,
            'resnet34' : resnet34
        }

    return image_sizes, metrics, batch_sizes, arch

def custom_configuration_0():
    image_sizes = [64, 128, 256, 512]
    predefined_metrics = [f2, average_precision, average_recall, hamming_score]
    batch_sizes = [8, 16, 32, 64]
    architectures = {
        'resnet34': resnet34, 
        'resnet50': resnet50, 
        'resnext50': resnext50
        }
    return image_sizes, predefined_metrics, batch_sizes, architectures

def custom_configuration_1():
    image_sizes = [64, 128, 256, 512]
    predefined_metrics = [f2, average_precision, average_recall, hamming_score]
    batch_sizes = [8, 16, 32, 64]
    architectures = {
        'resnext50': resnext50
        }
    return image_sizes, predefined_metrics, batch_sizes, architectures

def polar_configuration():
    image_sizes = [512]
    predefined_metrics = [f2, average_precision, average_recall, hamming_score]
    batch_sizes = [16]
    architectures = {
        'resnet34': resnet34
        }
    return image_sizes, predefined_metrics, batch_sizes, architectures

def parse_configuration(filename):
    input_file = open(filename, 'r').read().split('\n')
    image_sizes = list(map(int, input_file[0].split()))
    metrics = [getattr(sys.modules[__name__], metric_name) for metric_name in input_file[1].split()]
    batch_sizes = list(map(int, input_file[2].split()))
    architectures = {arch_name: getattr(sys.modules[__name__], arch_name) for arch_name in input_file[3].split()}
    return image_sizes, metrics, batch_sizes, architectures
    
if __name__ == '__main__':

    working_title = 'grid_search_test' if len(sys.argv) < 2 else sys.argv[1]
    gpu = 0 if len(sys.argv) < 3 else int(sys.argv[2])

    PATH = '/home/reflex/fastai/courses/dl1/data/reflex_multilabel'
    
    current_instance_dir = f'{PATH}/{working_title}'
    models_dir = f'{PATH}/{working_title}/models/'
    tmp_dir = f'{PATH}/{working_title}/tmp/'

    os.makedirs(current_instance_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    label_csv = f'{PATH}/train.csv' 
    val_idxs = list(range(1782, 1782 + 446 - 1)) 

    image_sizes, metrics, batch_sizes, architectures = custom_configuration_0()
    grid_search(PATH, label_csv, image_sizes, metrics, batch_sizes, architectures, val_idxs, working_title, gpu=gpu)


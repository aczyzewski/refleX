from __future__ import absolute_import
from celery import shared_task
import time

import sys
sys.path.insert(0, '/home/reflex/refleX/lib/fastai')
from fastai.conv_learner import *

@shared_task
def long_task(seconds=20):
    time.sleep(seconds)
    return [0.1,0.2,0.2,0.02,0.4,0.04,0.04]

@shared_task
def run_classifier(img_path):

    # *** PARAMETERS *** 
    model = resnet34
    image_size = 512
    # ***

    PATHS = {
        'CSV_PATH': '/home/reflex/refleX/metadata/6K/csv/fastai_labels_val_train.csv',
        'DATA_PATH': '/home/reflex/refleX/metadata/6K/',
        'DATA_FOLDER': 'original512',
    }

    transformations = tfms_from_model(model, image_size)
    data = ImageClassifierData.from_csv(
        PATHS['DATA_PATH'], PATHS['DATA_FOLDER'], PATHS['CSV_PATH'], 
        tfms=transformations, val_idxs=list(range(200)), test_name='original512', suffix='.png', bs=1
    )

    learner = ConvLearner.pretrained(model, data)
    learner.load('/home/reflex/refleX/results/fastai_experiments/resnet34_10012019/models/best_resnet34_imgsize512_batch16_unfreeze')

    trn_tfms, val_tfms = tfms_from_model(model, image_size)
    img = val_tfms(open_image(img_path))
    learner.precompute = False 

    return list(map(float, list(learner.predict_array(img[None])[0])))

    

"""
#TODO ! Finish dict
    results = {'status': True,

               'loop_scattering': 0,
               'background_ring': 0,
               'strong_background': 0,
               'diffuse_scattering': 0,
               'artifact': 0,
               'ice_ring': 0,
               'non_uniform_detector':0}
    return results
"""

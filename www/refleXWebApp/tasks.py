from __future__ import absolute_import
from celery import shared_task
import time
import sys
import os

try:
    sys.path.insert(0, '/home/reflex/refleX/lib/fastai')
    sys.path.insert(0, '/home/reflex/refleX/src')

    import cv2 as cv
    from fastai.conv_learner import *
    from rawdatalib import RawDataFile

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

        # TODO: TEST!
        if img_path.endswith('.npy.bz2'):
            img_file = RawDataFile(img_path)
            img = img_file.to_img(self, size=image_size)
            if img is None:
                return [0] * 7

            img_path = img_path[:-len('.npy.bz2')] + ".png"
            cv.imwrite(img_path, img)

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
        print(data.classes)

        os.remove(img_path)
        return [round(float(value), 3) for value in list(learner.predict_array(img[None])[0])]
except:
    print(" --- !!! FASTAI IS NOT SUPPORTED !!! --- ")
    @shared_task
    def long_task(seconds=20):
        time.sleep(seconds)
        return [0.1,0.6,0.2,0.02,0.8,0.04,0.04]

    @shared_task
    def run_classifier(img_path):
        time.sleep(10)
        os.remove(img_path)
        return [0.1,0.6,0.2,0.02,0.8,0.04,0.04]

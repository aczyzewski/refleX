from __future__ import absolute_import
from celery import shared_task
import time

@shared_task
def long_task(seconds=20):
    time.sleep(seconds)
    return [0,0,1,0,1,0,0]


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

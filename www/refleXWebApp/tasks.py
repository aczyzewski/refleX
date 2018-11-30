from __future__ import absolute_import
from celery import shared_task
import time

@shared_task
def long_task(seconds=20):
    time.sleep(seconds)
    #TODO ! Finish dict
    #results = {'status':
    #        'loop_scattering':}
    return True

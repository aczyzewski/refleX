from __future__ import absolute_import
from celery import shared_task
import time

@shared_task
def long_task(seconds=20):
    time.sleep(seconds)
    return True

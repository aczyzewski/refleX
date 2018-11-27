
from django.http import HttpResponse
from django.shortcuts import redirect, render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.template import loader

from .forms import ImageUploadForm
from .models import UserAdding

from .celery import app as celery_app
from .tasks import long_task
from celery.result import AsyncResult

import time
import random
import string
import os

# TODO: Ajax
# TODO: Celary

# --- HELPERS ---
def generate_random_hash(size):
    return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(size))

def save_uploaded_file(file_object):
    try:
        file_name, file_extension = os.path.splitext(file_object.name)
        dest_file_name = "%s_%s%s" % (file_name, generate_random_hash(8), file_extension)
        dest_file_path = os.path.join(settings.UPLOADS_URL, dest_file_name)
        with open(dest_file_path, 'wb') as dest_file:
            for chunk in file_object.chunks():
                dest_file.write(chunk)
    except:
        return False

    return True

# --- VIEWS ---
@login_required()
def index(request):
    # TODO: Merge redundant if-else cases.
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            if save_uploaded_file(request.FILES['picture']):

                long_task.delay(60)

                template = loader.get_template('refleXWebApp/loading.html')
                return HttpResponse(template.render({}, request))


                # TODO: Generate key!
                # TODO: Add task to the queue
            else:
                #FIXME: Error page: Cannot save the file!
                template = loader.get_template('refleXWebApp/success.html')
                return HttpResponse(template.render({}, request))
        else:
            # FIXME: Error page: Invalid form!
            template = loader.get_template('refleXWebApp/credits.html')
            return HttpResponse(template.render(context, request))
    else:
        context = {'form': ImageUploadForm()}
        template = loader.get_template('refleXWebApp/useradding_form.html')
        return HttpResponse(template.render(context, request))

def get_task_result(request, task_id):
    # 93abbdf4-117a-4a46-9d5d-234d718bdb79
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.ready():
        return HttpResponse('Task state: %s | Result: %s' % (task_result.state, task_result.get()))
    else:
        return HttpResponse("Not ready!")

def credits(request):
    template = loader.get_template('refleXWebApp/credits.html')
    return HttpResponse(template.render({}, request))
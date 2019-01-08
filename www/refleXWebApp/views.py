
from django.http import HttpResponse
from django.shortcuts import redirect, render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.template import loader

from .forms import ImageUploadForm
from .models import UserAdding, OutputScore

from .celery import app as celery_app
from .tasks import long_task
from celery.result import AsyncResult

import time
import random
import string
import os

from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.reverse import reverse

from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser
from .serializer import OutputScoreSerializer

# TODO: Ajax

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
#@login_required()
def index(request):
    # TODO: Merge redundant if-else cases.
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            if save_uploaded_file(request.FILES['picture']):

                task = long_task.delay()

                template = loader.get_template('refleXWebApp/loading.html')
                return HttpResponse(template.render({'task_id': task.id}, request))

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
        try:
            return HttpResponse('Task state: %s | Result: %s' % (task_result.state, task_result.get()))
        except:
            msg = ("77 views.py error")
            return HttpResponse(msg)
    else:
        return HttpResponse("Not ready!")

def credits(request):
    template = loader.get_stemplate('refleXWebApp/credits.html')
    return HttpResponse(template.render({}, request))

def return_loading(request):
    return render(request, 'refleXWebApp/loading.html')

@csrf_exempt
def api_list(request, task_id):
    if request.method == 'GET':
        task = AsyncResult(task_id, app=celery_app)
        result = [0] * 7
        if task.ready():
            result = [0] + [True] +  task.get()
        output = OutputScore(*result)
        serializer = OutputScoreSerializer(output)
        return JsonResponse(serializer.data, safe=False)

@csrf_exempt
def snippet_list(request):
    """
    List all code snippets, or create a new snippet.
    """
    if request.method == 'GET':
        snippets = OutputScore.objects.all()
        serializer = OutputScoreSerializer(snippets, many=True)
        return JsonResponse(serializer.data, safe=False)

    elif request.method == 'POST':
        data = JSONParser().parse(request)
        serializer = OutputScoreSerializer(data=data)
        if serializer.is_valid():
            serializer.save()
            return JsonResponse(serializer.data, status=201)
        return JsonResponse(serializer.errors, status=400)

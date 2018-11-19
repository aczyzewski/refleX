from django.shortcuts import render

from django.views.generic import CreateView
from django.views.generic import UpdateView

from mainPage.models import Person
from mainPage.forms import PersonForm, ImageUploadForm
from django.template import RequestContext
from django.http import Http404
from django.shortcuts import redirect, render, render_to_response

from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from mainPage.serializers import UserSerializer, GroupSerializer

def post_new(request):
    if request.method == "POST":
        form = PersonForm(request.POST)
        formimg = ImageUploadForm(request.POST, request.FILES)
        if all([form.is_valid(), formimg.is_valid()]):
            post = form.save(commit=False)
            post2 = formimg.save(commit=False)
            #handle_uploaded_file(request.POST['nameOfFile'],request.FILES['file'])
            response = redirect('/mainPage/loading')
            return response
    else:
        form = PersonForm()
        formimg = ImageUploadForm()

    return render(request, 'mainPage/form_page.html', {'form': form,'formimg': formimg})

def index(request):
    return render(request,'mainPage/index.html')

def success(request):
    return render(request,'mainPage/success.html')

def loading(request):
        return render(request,'mainPage/loading.html')

def credits(request):
        return render(request,'mainPage/credits.html')


def view404(request):
    return render(request, '404.html')

def view500(request):
    return render(request, '404.html')

#function responsible for image handling
def handle_uploaded_file(nameOFFile_,f):
    with open('static/uploadedPhotos/'+nameOFFile_, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return 0


class UserViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows users to be viewed or edited.
    """
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer


class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer

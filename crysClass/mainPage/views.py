from django.views.generic import ListView, CreateView, UpdateView
from .models import ExamineType,UserAdding
from mainPage.forms import ImageUploadForm
from django.template import RequestContext
from django.http import Http404,HttpResponseRedirect
from django.urls import reverse_lazy
from django.shortcuts import redirect, render, render_to_response

from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from .serializers import UserSerializer, GroupSerializer

from django.contrib.auth.models import User
from django.contrib import messages
from django.views.generic.edit import FormView
from .tasks import create_random_user_accounts

"""
class ClassCreateView(CreateView):
    model = UserAdding
    form_class = ImageUploadForm
    success_url = reverse_lazy('mainPage:loading')

class GenerateRandomUserView(FormView):
    template_name = 'mainPage/useradding_form.html'
    form_class = GenerateRandomUserForm

    def form_valid(self, form):
        total = form.cleaned_data.get('total')
        create_random_user_accounts.delay(total)
        messages.success(self.request, 'We are generating your random users! Wait a moment and refresh this page.')
        return redirect('users_list')
"""

def post_new(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST,request.FILES)
        if form.is_valid():
            m = UserAdding()
            m.examine_type = form.cleaned_data['examine_type']
            m.pic = form.cleaned_data['pic']
            m.save()

            #handle_uploaded_file(request.POST[''],request.FILES['pic'])
            return HttpResponseRedirect('success')
    else:
        form = ImageUploadForm()

    return render(request, 'mainPage/useradding_form.html', {'form': form})

#function responsible for image handling
def handle_uploaded_file(nameOFFile_,f):
    with open('uploadedPhotos/'+nameOFFile_, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

# just pages rendering ---------------------------------------------------
def index(request):
    return render(request,'mainPage/index.html')

def success(request):
    return render(request,'mainPage/success.html')

def loading(request):
        return render(request,'mainPage/loading.html')

def credits(request):
        return render(request,'mainPage/credits.html')

#error handlers
def view404(request):
    return render(request, '404.html')

def view500(request):
    return render(request, '404.html')

# API part ----------------------------------------------------------
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

from mainPage.forms import ImageUploadForm
from django.template import RequestContext
from django.template import RequestContext
from django.http import Http404,HttpResponseRedirect
from django.urls import reverse_lazy
from django.shortcuts import redirect, render, render_to_response

from django.conf import settings
from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from .serializers import UserSerializer, GroupSerializer

from django.contrib.auth.models import User
import os

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

#function responsible for image handling
def handle_uploaded_file(nameOFFile_,f):
    #fixme with open(...) optimize '+' signs
    with open(settings.BASE_DIR + '/mainPage/test_photos/' + nameOFFile_ + '.jpg','wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def post_new(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST,request.FILES)
        if form.is_valid():
            handle_uploaded_file(request.POST['examine_type'],request.FILES['pic'])
            return HttpResponseRedirect('success')
    else:
        ImageUploadForm()
    return render(request, 'mainPage/useradding_form.html', {'form': ImageUploadForm()})


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

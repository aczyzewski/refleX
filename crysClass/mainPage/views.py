from . import forms
from django.shortcuts import render

from django.views.generic import CreateView
from django.views.generic import UpdateView

from mainPage.models import Person
from mainPage.forms import PersonForm
from django.http import Http404

class PersonCreateView(CreateView):
    model = Person
    #fields = ('name of file', 'your Email', 'Link To File')
    fields = ('nameOfFile', 'yourEmail', 'linkToFile')
    template_name = 'mainPage/person_form.html'

def index(request):
    return render(request,'mainPage/index.html')

def success(request):
    return render(request,'mainPage/success.html')

def loading(request):
        return render(request,'mainPage/progress_bar.html')

def credits(request):
        return render(request,'mainPage/credits.html')

"""
tut (3.3) 137?
def form_name_view(request):
    form = forms.FormName()

    if request.method == 'POST':
        form = forms.FormName(request.POST)

        if form.is_valid():
            # DO STH
            print("VALIDATION SUCCESSED !")
            print("NAME:"+ form.cleaned_data['name'])
            print("EMAIL:"+ form.cleaned_data['email'])

    return render(request,'mainPage/form_page.html',{'form':form})
"""


"""
def handler404(request, *args, **argv):
    response = render_to_response('404.html', {},
                                  context_instance=RequestContext(request))
    response.status_code = 404
    return response


def handler500(request, *args, **argv):
    response = render_to_response('500.html', {},
                                  context_instance=RequestContext(request))
    response.status_code = 500
    return response
"""

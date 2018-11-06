from django.conf.urls import url
from mainPage import views
from django.urls import path

from mainPage.models import Person
from mainPage.views import PersonCreateView

app_name =  'mainPage'

urlpatterns = [
    url('add', PersonCreateView.as_view(model=Person, success_url="loading")),
    url('success', views.success, name='success'),
    url('credits', views.credits, name='credits'),
    url('loading', views.loading, name='loading'),
]

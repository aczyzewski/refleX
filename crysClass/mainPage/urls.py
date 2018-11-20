from django.conf.urls import url
from . import views
from django.urls import path,include
from django.conf.urls.static import static
from .models import Person
from django.conf import settings

app_name =  'mainPage'


urlpatterns = [
    url('add', views.post_new, name='add'),
    #url('add/', views.ClassCreateView.as_view(), name='add'),

    url('loading', views.loading, name='loading'),
    url('success', views.success, name='success'),
    url('credits', views.credits, name='credits'),
]

"""crysClass URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include

from mainPage import views
from mainPage.models import Person

from django.conf.urls import url, include
from rest_framework import routers

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'groups', views.GroupViewSet)



urlpatterns = [
    #path('add/', PersonCreateView.as_view(model=Person, success_url="\success")),
    #url('formpage/',views.form_name_view, name='form_name'),
    #url(r'success', views.success, name='success'),

    url(r'', include(router.urls)),
    url(r'api-auth/', include('rest_framework.urls', namespace='rest_framework')),

    url(r'^$',views.index, name='index'),
    url(r'admin/', admin.site.urls, name='admin'),
    url(r'mainPage/', include('mainPage.urls')),
]


handler404 = 'mainPage.views.view404'
handler500 = 'mainPage.views.view500'

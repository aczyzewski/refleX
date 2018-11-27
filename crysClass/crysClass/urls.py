from django.contrib import admin
from django.urls import path
from django.conf.urls import url, include
from mainPage import views

from django.conf.urls import url, include

from rest_framework import routers

router = routers.DefaultRouter()
router.register(r'users', views.UserViewSet)
router.register(r'groups', views.GroupViewSet)

urlpatterns = [
    #url(r'^$',views.index, name='index'),
    url(r'', include(router.urls)),
    url(r'admin/', admin.site.urls, name='admin'),
    url(r'mainPage/', include('mainPage.urls')),

    url(r'api-auth/', include('rest_framework.urls', namespace='rest_framework')),

]

handler404 = 'mainPage.views.view404'
handler500 = 'mainPage.views.view500'

from django.urls import re_path, path, include
from . import views

urlpatterns = [
    re_path(r'^index\.*', views.index),                     # TODO: one explicit regex!
    path('', views.index, name="index"),                 # TODO: one explicit regex!
    path('credits/', views.credits, name="credits"),
    path('accounts/', include('django.contrib.auth.urls'))
]

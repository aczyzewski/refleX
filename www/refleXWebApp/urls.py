from django.urls import re_path, path, include
from . import views

urlpatterns = [
    re_path(r'^index\.*', views.index),                     # TODO: one explicit regex!
    path('', views.index, name="index"),                 # TODO: one explicit regex!

    # API
    # path('api/', ...)
    path('loading/', views.return_loading),
    path('result/<str:task_id>', views.get_task_result),
    path('credits/', views.credits, name="credits"),
    path('accounts/', include('django.contrib.auth.urls')),

    path('api/result/<str:task_id>', views.api_list),

    path('snippets/', views.snippet_list),
    #path('snippets/<int:pk>/', views.snippet_detail),
]

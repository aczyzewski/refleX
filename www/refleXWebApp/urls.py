from django.urls import re_path, path, include
from . import views

urlpatterns = [
    re_path(r'^index\.*', views.index),                     # TODO: one explicit regex!
    path('', views.index, name="index"),                 # TODO: one explicit regex!

    # API
    # path('api/', ...)
    path('credits/', views.return_credits, name="credits"),
    path('results/', views.return_results, name="results"),

    path('result/<str:task_id>', views.get_task_result),
    path('api/result/<str:task_id>', views.api_list),

    path('accounts/', include('django.contrib.auth.urls')),
    path('snippets/', views.snippet_list),
    #path('snippets/<int:pk>/', views.snippet_detail),
]

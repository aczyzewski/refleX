from django.views.generic.base import RedirectView
from django.urls import re_path, path, include
from django.contrib.staticfiles.storage import staticfiles_storage
from django.conf.urls import url
from . import views

urlpatterns = [
    re_path(r'^index\.*', views.index),                     # TODO: one explicit regex!
    path('', views.index, name="index"),                 # TODO: one explicit regex!

    # API
    # path('api/', ...)
    path('credits/', views.return_credits, name="credits"),

    url('favicon.ico$',
        RedirectView.as_view( # the redirecting function
            url=staticfiles_storage.url('favicon.ico'), # converts the static directory + our favicon into a URL
            # in my case, the result would be http://www.tumblingprogrammer.com/static/img/favicon.ico
        ),
        name="favicon" # name of our view
    ),

    path('result/<str:task_id>', views.get_task_result),
    path('api/result/<str:task_id>', views.api_list),

    path('accounts/', include('django.contrib.auth.urls')),
    path('snippets/', views.snippet_list),
    #path('snippets/<int:pk>/', views.snippet_detail),
]

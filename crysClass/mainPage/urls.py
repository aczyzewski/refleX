from django.conf.urls import url
from mainPage import views
from django.urls import path
from django.conf.urls.static import static
from mainPage.models import Person
from django.conf import settings

app_name =  'mainPage'


urlpatterns = [
    #url('',views.post_new, name='post_new'),
    url('add', views.post_new, name='post_new'),
    url('loading', views.loading, name='loading'),
    url('success', views.success, name='success'),
    url('credits', views.credits, name='credits'),
]
#+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

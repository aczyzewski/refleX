from django.db import models
from django.utils.translation import ugettext_lazy as _

class Person(models.Model):
    nameOfFile = models.CharField(max_length=130)
    yourEmail = models.EmailField(blank=False)
    linkToFile = models.CharField(max_length=50, blank=True)

class Img(models.Model):
    pic = models.ImageField(upload_to = 'static/getPhotos', default = 'static/getPhotos/None/no-img.jpg')

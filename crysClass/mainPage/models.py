from django.db import models
from django.utils.translation import ugettext_lazy as _

class Person(models.Model):
    nameOfFile = models.CharField(max_length=130)
    yourEmail = models.EmailField(blank=False)
    linkToFile = models.CharField(max_length=50, blank=True)

class Img(models.Model):
    pic = models.ImageField(upload_to = 'static/getPhotos', default = 'static/getPhotos/None/no-img.jpg')

class ExamineType(models.Model):
    examine_type = models.CharField(max_length=120, blank=False)

    def __str__(self):
        return self.examine_type

    def __unicode__(self):
        return smart_unicode(self.customer_type)

class UserAdding(models.Model):
    examine_type = models.ForeignKey(ExamineType, on_delete='cascade')
    pic = models.ImageField()

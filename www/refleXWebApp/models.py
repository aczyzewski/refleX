from django.db import models
from django.utils.translation import ugettext_lazy as _
from django.contrib.auth.models import User

class UserAdding(models.Model):
    picture = models.ImageField()

# class for API results
class OutputScore(models.Model):
    status = models.BooleanField()
    artifact = models.FloatField(default=0, max_length=10)
    background_ring = models.FloatField(default=0, max_length=10)
    diffuse_scattering = models.FloatField(default=0, max_length=10)
    ice_ring = models.FloatField(default=0, max_length=10)
    loop_scattering =  models.FloatField(default=0, max_length=10)
    non_uniform_detector = models.FloatField(default=0, max_length=10)
    strong_background = models.FloatField(default=0, max_length=10)

    def __str__(self):
            return str(self.id)

# check if keys is in db
class Keys(models.Model):
    key = models.IntegerField();
from django.db import models

class Person(models.Model):
    #fields = ('Name of file', 'Your Email', 'Link to file')
    nameOfFile = models.CharField(max_length=130)
    yourEmail = models.EmailField(blank=True)
    linkToFile = models.CharField(max_length=50, blank=False)

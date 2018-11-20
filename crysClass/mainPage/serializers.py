from django.contrib.auth.models import User, Group
from rest_framework import serializers
from .models import UserAdding

class UserAddingSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('examine_type', 'pic')

class UserSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = User
        fields = ('url', 'username', 'email', 'groups')


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ('url', 'name')

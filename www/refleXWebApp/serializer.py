from .models import OutputScore
from rest_framework import serializers

class OutputScoreSerializer(serializers.Serializer):
    status = serializers.BooleanField()
    loop_scattering = serializers.FloatField(default=0)
    background_ring = serializers.FloatField(default=0)
    strong_background = serializers.FloatField(default=0)
    diffuse_scattering = serializers.FloatField(default=0)
    artifact = serializers.FloatField(default=0)
    ice_ring = serializers.FloatField(default=0)
    non_uniform_detector = serializers.FloatField(default=0)

    def create(self, validated_data):
        """
        Create and return a new `Snippet` instance, given the validated data.
        """
        return OutputScore.objects.create(**validated_data)

    def update(self, instance, validated_data):
        """
        Update and return an existing `Snippet` instance, given the validated data.
        """
        instance.status = validated_data.get('status', instance.status)
        instance.loop_scattering = validated_data.get('loop_scattering', instance.loop_scattering)
        instance.background_ring = validated_data.get('background_ring', instance.background_ring)
        instance.strong_background = validated_data.get('strong_background', instance.strong_background)
        instance.diffuse_scattering = validated_data.get('diffuse_scattering', instance.diffuse_scattering)
        instance.artifact = validated_data.get('artifact', instance.artifact)
        instance.ice_ring = validated_data.get('ice_ring', instance.ice_ring)
        instance.non_uniform_detector = validated_data.get('non_uniform_detector', instance.non_uniform_detector)
        instance.save()
        return instance

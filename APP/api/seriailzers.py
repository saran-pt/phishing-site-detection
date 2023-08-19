from .models import Urls
from rest_framework import serializers

class UrlSerializer(serializers.ModelSerializer):
    class Meta:
        model = Urls
        fields = '__all__'
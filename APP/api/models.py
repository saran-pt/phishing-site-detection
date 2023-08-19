from django.db import models

# Create your models here.
class Urls(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    url = models.CharField(max_length=600)
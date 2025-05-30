from django.db import models

# Create your models here.
class DetectFile(models.Model):
    file = models.FileField(upload_to='images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
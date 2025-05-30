from django import forms
from .models import DetectFile

class FileUploadForm(forms.ModelForm):
    class Meta:
        model = DetectFile
        fields = ['file']

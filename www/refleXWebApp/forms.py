from django.forms import ModelForm
from .models import UserAdding

class ImageUploadForm(ModelForm):
    class Meta:
        model = UserAdding
        fields = ['picture']
        labels = {'picture': 'Load image:'}

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

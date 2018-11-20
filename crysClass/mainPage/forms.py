from django.forms import ModelForm
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit
from .models import UserAdding, ExamineType
from django.utils.translation import gettext_lazy as _

class ImageUploadForm(ModelForm):
    class Meta:
        model = UserAdding
        fields = ('examine_type','pic')
        labels = {'examine_type': '1. Choose type of reserach: ' ,'pic':'2. Load image: '}

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fields['examine_type'].queryset = ExamineType.objects.none()

# tutorial lesson 137 (3.3)

"""
class FormName(forms.Form):
    name = forms.CharField()
    email = forms.EmailField()
    text = forms.CharField(widget=forms.Textarea)
    botcatcher = forms.CharField(required=False,
                                widget=forms.HiddenInput)

    def clean_botcatcher(self):
        botcatcher = self.cleaned_data['botchacher']
        if len(botcatcher) > 0:
            raise forms.validationError()
"""

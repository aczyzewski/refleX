from django.forms import ModelForm
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit
from mainPage.models import Person, Img
from django.utils.translation import gettext_lazy as _

class PersonForm(ModelForm):
    class Meta:
        model = Person
        fields = ('nameOfFile', 'yourEmail','linkToFile')
        labels = {'nameOfFile': 'Name of File:', 'yourEmail':'Email:','linkToFile':'Company:'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Save person'))


class ImageUploadForm(ModelForm):
    class Meta:
        model = Img
        fields = ('pic',)
        labels = {'pic':'Load image:'}

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

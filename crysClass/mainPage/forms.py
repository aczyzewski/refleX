from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit
from mainPage.models import Person


class PersonForm(forms.ModelForm):
    class Meta:
        model = Person
        #fields = ('name', '','linkToFile')
        fields = ('nameOfFile', 'yourEmail','linkToFile')
        #fields = ('nameOfFile', 'yourEmail','linkToFile')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Run'))


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

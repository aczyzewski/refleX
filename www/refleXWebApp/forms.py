from django import forms

class ImageUploadForm(forms.Form):
    picture = forms.FileField(
        label = 'Choose a file...',
        required = True,
        widget = forms.FileInput(
            attrs = {
                'id' : "file",
                'class' : "inputfile",
                'onchange' : "$('#uploadForm label').html(this.files[0].name)"
                }
        )
    )
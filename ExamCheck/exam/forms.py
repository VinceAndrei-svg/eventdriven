from django import forms
from .models import AnswerSheet

class AnswerSheetForm(forms.ModelForm):
    class Meta:
        model = AnswerSheet
        fields = ['file']

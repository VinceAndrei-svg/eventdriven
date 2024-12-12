from django.urls import path
from .views import upload_sheet

urlpatterns = [
    path('', upload_sheet, name='upload_sheet'),
]

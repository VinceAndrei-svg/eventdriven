import os
from django.shortcuts import render
from .forms import UploadImageForm
from ml_model.bubble_classifier import predict_answer

def upload_image(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']
            image_path = os.path.join('bubble_sheet/static/uploads', image.name)
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)
            answer = predict_answer(image_path)
            return render(request, 'bubble_sheet/upload.html', {'form': form, 'answer': answer})
    else:
        form = UploadImageForm()
    return render(request, 'bubble_sheet/upload.html', {'form': form})

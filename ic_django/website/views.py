from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from .model_and_preprocessing.test_model import process_single_image, os

def home(request):
    if not os.path.exists('temp'):
       os.makedirs('temp')
    if request.method == 'POST':
        # Handle image upload
        image = request.FILES.get('image')
        path = '/model_and_preprocessing/Experimental'
        if image:
            # Save the uploaded image to a temporary location
            temp_image_path = f"temp/{image.name}"
            with open(temp_image_path, 'wb') as temp_image:
                for chunk in image.chunks():
                    temp_image.write(chunk)

            # Process the uploaded image
            prediction = process_single_image(temp_image_path)

            # Delete the temporary image file
            os.remove(temp_image_path)
    
            return render(request, 'home.html', {'prediction': prediction})

    return render(request, 'home.html', {})


def get_prediction(path):
    prediction = process_single_image(path)
    return prediction

def logout_user():
    pass

def register_user():
    pass

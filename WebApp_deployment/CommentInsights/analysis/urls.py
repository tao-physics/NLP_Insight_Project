from django.urls import path
from analysis import views

urlpatterns = [
    path('home/', views.homepage, name='home'),
    path('upload/', views.upload_file, name='upload'),
    path('result/', views.prediction_result, name='result'),
]
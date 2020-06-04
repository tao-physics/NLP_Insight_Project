from django.urls import path
from analysis import views

urlpatterns = [
    path('home/', views.homepage, name='home'),
]
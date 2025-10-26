from django.urls import path
from . import views

urlpatterns = [
    path('statistical/', views.statistical_page, name='statistical'),
]
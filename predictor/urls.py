from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_page, name='predict'),
    path('api/predict/', views.predict_request, name='predict_request'),
]
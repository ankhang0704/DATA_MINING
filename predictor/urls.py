from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_page, name='predict'),
    path('api/predict/', views.predict_request, name='predict_request'),
    path('unemployment/', views.predict_unemployment_page, name='unemployment'),
    path('api/predict_unemployment/', views.predict_unemployment_request, name='predict_unemployment_request'),
]
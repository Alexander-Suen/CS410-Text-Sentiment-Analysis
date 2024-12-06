from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('test/', views.scrape_reviews, name='test'),
    path('check_model_status/', views.check_model_status, name='check_model_status'),
]
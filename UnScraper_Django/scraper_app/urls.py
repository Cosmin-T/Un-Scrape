# scraper_app/urls.py

from django.urls import path
from . import views

# Remove the app_name line
urlpatterns = [
    path('', views.scrape_website, name='index'),
]
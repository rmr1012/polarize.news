from django.urls import path, re_path
from home.views import  *
from . import views

urlpatterns = [
    path('', HomeView.as_view(), name='home'),  #as_view is methode in TemplateView
    path('load', LoadAPI, name='home'),  #loading end point, return 1 pair
    path('search', SearchAPI, name='home'), #loading end point, return 3 pair
    path('fetch', FetchAPI, name='home'), #loading end point, return 3 pair 
]

from django.urls import path, re_path
from home.views import  *
from . import views

urlpatterns = [
    path('', HomeView.as_view(), name='home'),  #as_view is methode in TemplateView

]

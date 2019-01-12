from django.views.generic import TemplateView
from django.shortcuts import render, redirect

from django.urls import reverse


class HomeView(TemplateView): #some from 48
    template_name = 'home/index.html'
    def get(self, request):
        return render(request, self.template_name)
    def post(self, request):
        return render(request, self.template_name)

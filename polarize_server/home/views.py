from django.views.generic import TemplateView, View
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

from django.urls import reverse
from home.testCard import *

class HomeView(TemplateView): #some from 48
    template_name = 'home/index.html'
    def get(self, request):
        context = {'context':dummyContext}
        return render(request, self.template_name,context)
    def post(self, request):
        return render(request, self.template_name,context)

@api_view(['GET','POST'])
@csrf_exempt
def LoadAPI(request):
    if request.method == 'GET':
        print("haha")
        return render(request, 'home/index.html')
    else:
        print("hehe")
        return render(request, 'home/index.html')

@api_view(['GET','POST'])
@csrf_exempt
def SearchAPI(request):
    if request.method == 'GET':
        print("haha")
        return render(request, 'home/index.html')
    else:
        print("hehe")
        return render(request, 'home/index.html')

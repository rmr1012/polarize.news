from django.views.generic import TemplateView, View
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.http import HttpResponseForbidden , HttpResponse
from django.urls import reverse
from home.testCard import *
from home.models import *
from home.algorithms import *

topWords= ["trump","president","government","shutdown","border","donald","christmas","washington","police","national","democrats","republicans","syria","security","senate","china","california","election","war","school","pelosi","campaign"]

class HomeView(TemplateView): #some from 48
    template_name = 'home/index.html'
    def get(self, request):
        keyword="headlines"
        # queryset = CardRackCache.objects.filter(keyword=keyword).order_by('-timestamp')
        # if(queryset): # if there's a recent copy avail, grab that
        #     context = {'context':queryset[0]['jsonStr']}
        realContext =  get_headlines(page_size=100, sources=relevant_sources_str)

        print(realContext)
        print(dummyContext)
        context = {'context':realContext} # delete dummy when there's real stuff
        return render(request, self.template_name,context)
    def post(self, request):
        return render(request, self.template_name,context)

@api_view(['GET','POST'])
@csrf_exempt
def LoadAPI(request):
    #print(request.POST)
    if request.method == 'GET':
        print("haha")
        return render(request, 'home/index.html')
    else: #POST
        #try:
        print("step1")
        index=request.POST['index']
        topic=request.POST['topic']
        print(topic,index)
        # hit DB here, but for now render dummy
        context = {"context":[dummyContext[0]]}
        print("step2")

        # return render(request, 'wizzard/wizzard.html',context)
        newCard = render_to_string('home/load_one_row.html', context)
        print("step3")

        print(type(newCard))
        print(newCard)
        print("step4")
        return JsonResponse({"card":newCard})

        # except Exception as e:
        #     print(str(e))
        #     print("throwing 403")
        #     return HttpResponseForbidden()

@api_view(['GET','POST'])
@csrf_exempt
def SearchAPI(request):
    if request.method == 'GET':
        print("haha")
        return render(request, 'home/index.html')
    else:
        print("hehe")
        return render(request, 'home/index.html')

@api_view(['GET','POST'])
@csrf_exempt
def FetchAPI(request):
    if request.method == 'GET':
        print(request.GET)
    return HttpResponse('')

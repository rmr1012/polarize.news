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
from random import *
topWords= ["donald trump","goverment shutdown","muller investigation","travel","border","immigration","abortion","supreme court","police","south china sea","democrats","republicans","syria","mexico","senate","white house","california","election","africa","education","nancy pelosi","campaign"]

class HomeView(TemplateView): #some from 48
    template_name = 'home/index.html'
    def get(self, request):
        firstChoices=sample(range(0,len(topWords)-1 ), 3)

        # queryset = CardRackCache.objects.filter(keyword=keyword).order_by('-timestamp')
        # if(queryset): # if there's a recent copy avail, grab that
        #     context = {'context':queryset[0]['jsonStr']}
        realContext1=get_headlines(topWords[firstChoices[0]], page_size=100, sources=relevant_sources_str)
        realContext2=get_headlines(topWords[firstChoices[1]], page_size=100, sources=relevant_sources_str)
        realContext3=get_headlines(topWords[firstChoices[2]], page_size=100, sources=relevant_sources_str)

        #realContext.append(get_headlines('government shutdown', page_size=100, sources=relevant_sources_str))
        #print(type(realContext))
        context = {'context':[realContext1,realContext2,realContext3],'topic':[topWords[firstChoices[0]],topWords[firstChoices[1]],topWords[firstChoices[2]]]} # delete dummy when there's real stuff
        return render(request, self.template_name,context)
    def post(self, request):
        return render(request, self.template_name,context)

@api_view(['GET','POST'])
@csrf_exempt
def LoadAPI(request):
    #print(request.POST)
    if request.method == 'GET':
        return render(request, 'home/index.html')
    else: #POST
        #try:
        print("step1")
        #print(request.data)
        topic=request.POST.get('topic[]')

        #print(topic,index)
        avalWords=[]
        for word in topWords:
            if word not in topic:
                avalWords.append(word)

        freeInd=sample(range(0,len(avalWords)-1 ), 1)[0]
        print(avalWords)

        keyword=avalWords[freeInd]
        print(keyword)
        realContext=get_headlines(keyword, page_size=100, sources=relevant_sources_str)

        context = {"context":realContext,"topic":keyword}
        print("step2")
        # return render(request, 'wizzard/wizzard.html',context)
        newCard = render_to_string('home/load_one_row.html', context)
        print("step3")

        print(type(newCard))
        #print(newCard)
        print("step4")
        return JsonResponse({"card":newCard})

        # except Exception as e:
        #     print(str(e))
        #     print("throwing 403")
        #     return HttpResponseForbidden()

@api_view(['GET','POST'])
@csrf_exempt
def SearchAPI(request):
    #print(request.POST)
    if request.method == 'GET':
        return render(request, 'home/index.html')
    else: #POST
        #try:
        print("step1")
        #print(request.data)
        topic=request.POST.get('topic[]')
        inquery=request.POST.get('query')
        #print(topic,index)
        avalWords=[]
        for word in topWords:
            if word not in topic:
                avalWords.append(word)

        realContext=get_headlines(inquery, page_size=100, sources=relevant_sources_str)

        context = {"context":realContext,"topic":inquery}
        print("step2")
        # return render(request, 'wizzard/wizzard.html',context)
        newCard = render_to_string('home/load_one_row.html', context)
        print("step3")

        print(type(newCard))
        #print(newCard)
        print("step4")
        return JsonResponse({"card":newCard})

        # except Exception as e:
        #     print(str(e))
        #     print("throwing 403")
        #     return HttpResponseForbidden()

@api_view(['GET','POST'])
@csrf_exempt
def FetchAPI(request):
    if request.method == 'GET':
        print(request.GET)
    return HttpResponse('')

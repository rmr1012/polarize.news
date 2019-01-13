from django.db import models

# Create your models here.
class CardRackCache(models.Model): # permit
    keyword = models.CharField(primary_key=True ,max_length=50,null=False) # headline, query(eg. Trump, shits)
    timestamp = models.DateTimeField(auto_now_add=True,null=False)
    sourceHash = models.CharField(max_length=50,null=False)
    jsonStr = models.CharField(max_length=65536,null=False)

# class NewsAPIAccessRecord(mdoels.Model):
#     timestamp = models.DateTimeField(auto_now_add=True,null=False)
#     keyword = models.CharField(max_length=50,null=False) # headline, query(eg. Trump, shits)

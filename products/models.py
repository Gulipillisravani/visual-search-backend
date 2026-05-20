from django.db import models


class Product(models.Model):

    name = models.CharField(max_length=500)
    category = models.CharField(max_length=200, blank=True, null=True)
    subcategory = models.CharField(max_length=200, blank=True, null=True)
    color = models.CharField(max_length=100, blank=True, null=True)
    gender = models.CharField(max_length=100, blank=True, null=True)
    season = models.CharField(max_length=100, blank=True, null=True)
    usage = models.CharField(max_length=100, blank=True, null=True)

    price = models.IntegerField(default=0)

    # IMPORTANT FIX: store image path properly
    image = models.ImageField(upload_to='dataset/images')

    def __str__(self):
        return self.name
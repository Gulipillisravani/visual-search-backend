from django.urls import path
from .views import search_products, get_products

urlpatterns = [

    path('search/', search_products),

    path('products/', get_products),

]
from django.urls import path

from .views import (
    get_products,
    image_search
)

urlpatterns = [

    path('', get_products),

    path(
        'search/',
        image_search
    ),
]
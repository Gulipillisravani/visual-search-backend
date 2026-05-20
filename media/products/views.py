from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Product
from .serializers import ProductSerializer


@api_view(['GET'])
def get_products(request):

    products = Product.objects.all()

    serializer = ProductSerializer(
        products,
        many=True
    )

    return Response(serializer.data)


@api_view(['POST'])
def image_search(request):

    image = request.FILES['image']

    return Response({
        "message": "Image received successfully"
    })


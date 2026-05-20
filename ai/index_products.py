import os
import django

# ADD THIS
os.environ.setdefault(
    'DJANGO_SETTINGS_MODULE',
    'core.settings'
)

django.setup()

from products.models import Product

from ai.clip_model import generate_embedding
from ai.search import add_embedding


products = Product.objects.all()


for product in products:

    try:

        image_path = product.image.path

        embedding = generate_embedding(
            image_path
        )

        add_embedding(
            embedding,
            product.id
        )

        print(
            f"Indexed: {product.name}"
        )

    except Exception as e:

        print(
            f"Error indexing {product.name}: {e}"
        )


print("All products indexed")
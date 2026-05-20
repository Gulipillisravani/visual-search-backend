import os
import numpy as np
import faiss

from rest_framework.decorators import api_view
from rest_framework.response import Response

from products.models import Product
from products.serializers import ProductSerializer

from ai.clip_model import generate_embedding

# -----------------------------------
# LOAD FAISS INDEX
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

index = faiss.read_index(
    os.path.join(BASE_DIR, "product_index.faiss")
)

product_ids = np.load(
    os.path.join(BASE_DIR, "product_ids.npy")
)

# -----------------------------------
# SEARCH PRODUCTS
# -----------------------------------

@api_view(['POST'])
def search_products(request):

    try:

        # -----------------------------
        # GET IMAGE + TEXT
        # -----------------------------

        uploaded_image = request.FILES.get("image")

        search_text = request.data.get(
            "search_text",
            ""
        ).lower()

        # -----------------------------
        # BOTH EMPTY
        # -----------------------------

        if not uploaded_image and not search_text:

            return Response({
                "error": "Upload image or enter search text"
            }, status=400)

        # =================================================
        # TEXT SEARCH ONLY
        # =================================================

        if not uploaded_image and search_text:

            matched_products = Product.objects.filter(
                category__icontains=search_text
            ) | Product.objects.filter(
                color__icontains=search_text
            ) | Product.objects.filter(
                gender__icontains=search_text
            )

            serializer = ProductSerializer(
                matched_products.distinct(),
                many=True,
                context={"request": request}
            )

            return Response(serializer.data)

        # =================================================
        # IMAGE SEARCH
        # =================================================

        # SAVE TEMP IMAGE

        temp_image_path = os.path.join(
            BASE_DIR,
            "temp.jpg"
        )

        with open(temp_image_path, "wb+") as f:

            for chunk in uploaded_image.chunks():
                f.write(chunk)

        # GENERATE EMBEDDING

        query_embedding = generate_embedding(
            temp_image_path
        )

        if query_embedding is None:

            return Response({
                "error": "Embedding generation failed"
            }, status=500)

        # NORMALIZE

        query_embedding = query_embedding.astype(
            "float32"
        )

        # SEARCH

        k = 100

        distances, indices = index.search(
            query_embedding,
            k
        )

        matched_products = []

        for idx in indices[0]:

            if idx < len(product_ids):

                product_id = int(
                    product_ids[idx]
                )

                try:

                    product = Product.objects.get(
                        id=product_id
                    )

                    # ---------------------------------
                    # IMAGE + TEXT FILTER
                    # ---------------------------------

                    if search_text:

                        text = search_text.lower()

                        combined_text = f"""
                        {product.category}
                        {product.color}
                        {product.gender}
                        """.lower()

                        words = text.split()

                        if any(
                            word in combined_text
                            for word in words
                        ):
                            matched_products.append(
                                product
                            )

                    else:

                        matched_products.append(
                            product
                        )

                except Product.DoesNotExist:
                    pass

        # REMOVE DUPLICATES

        unique_products = []

        added_ids = set()

        for product in matched_products:

            if product.id not in added_ids:

                unique_products.append(product)

                added_ids.add(product.id)

        # SERIALIZE

        serializer = ProductSerializer(
            unique_products,
            many=True,
            context={"request": request}
        )

        return Response(serializer.data)

    except Exception as e:

        print("SEARCH ERROR:", e)

        return Response({
            "error": str(e)
        }, status=500)
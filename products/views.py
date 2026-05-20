from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Product

from PIL import Image

import torch
import open_clip
import faiss
import pickle
import numpy as np


# DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD CLIP MODEL
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)

model.to(device)

# LOAD FAISS INDEX
index = faiss.read_index("faiss_index.bin")

# LOAD PRODUCT DATA
with open("products.pkl", "rb") as f:
    products_data = pickle.load(f)

@api_view(["POST"])
def search_products(request):

    try:

        image = request.FILES.get("image")

        search_text = request.data.get(
            "search_text",
            ""
        ).lower()

        if not image:

            return Response({
                "error": "No image uploaded"
            }, status=400)

        # OPEN IMAGE
        image_pil = Image.open(
            image
        ).convert("RGB")

        # PREPROCESS
        image_input = preprocess(
            image_pil
        ).unsqueeze(0).to(device)

        # EXTRACT FEATURES
        with torch.no_grad():

            image_features = (
                model.encode_image(
                    image_input
                )
            )

        # NORMALIZE
        image_features /= (
            image_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        query_embedding = (
            image_features
            .cpu()
            .numpy()
            .astype("float32")
        )

        # SEARCH
        distances, indices = (
            index.search(
                query_embedding,
                100
            )
        )

        results = []

        for idx, score in zip(
            indices[0],
            distances[0]
        ):

            if idx >= len(products_data):
                continue

            item = products_data[idx]

            try:

                product = Product.objects.get(
                    id=item["id"]
                )

            except:
                continue

            # TEXT FILTER
            if search_text:

                combined = f"""
                {product.category}
                {product.subcategory}
                {product.color}
                """.lower()

                if search_text not in combined:
                    continue

            image_url = ""

            try:

                if product.image:

                    image_url = (
                        request.build_absolute_uri(
                            product.image.url
                        )
                    )

            except:
                pass

            results.append({

                "id": product.id,

                "name": product.name,

                "category": product.category,

                "subcategory": product.subcategory,

                "color": product.color,

                "gender": product.gender,

                "season": product.season,

                "usage": product.usage,

                "price": product.price,

                "image": image_url,

                "images": [
                    image_url,
                    image_url,
                    image_url
                ],

                "similarity": round(
                    float(score),
                    2
                ),

                "offer": 20,

                "rating": 4.5,

                "detected_color":
                    product.color,

                "pattern": "Modern",

                "style": "Casual",

                "detected_category":
                    product.category
            })

        return Response(results)

    except Exception as e:

        print("SEARCH ERROR:", e)

        return Response({
            "error": str(e)
        }, status=500)


@api_view(["GET"])
def get_products(request):

    try:

        products = Product.objects.all()

        results = []

        for product in products:

            image_url = ""

            try:

                if product.image:

                    image_url = request.build_absolute_uri(
                        product.image.url
                    )

            except:
                pass

            results.append({

                "id": product.id,

                "name": product.name,

                "category": product.category,

                "subcategory": product.subcategory,

                "color": product.color,

                "gender": product.gender,

                "season": product.season,

                "usage": product.usage,

                "price": product.price,

                "image": image_url,

                "images": [
                    image_url,
                    image_url,
                    image_url
                ]
            })

        return Response(results)

    except Exception as e:

        print("GET ERROR:", e)

        return Response({
            "error": str(e)
        }, status=500)
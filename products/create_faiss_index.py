import os
import sys
import django
import pickle
import numpy as np
import faiss
import torch
import open_clip

from PIL import Image


print("STARTED")


# DJANGO SETUP
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "core.settings"
)

django.setup()

print("DJANGO READY")


from products.models import Product

print("PRODUCT MODEL IMPORTED")


# DEVICE
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("DEVICE:", device)


# LOAD MODEL
print("LOADING MODEL...")

model, _, preprocess = (
    open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai"
    )
)

model.to(device)

print("MODEL LOADED")


# GET PRODUCTS
products = Product.objects.all()

print("TOTAL PRODUCTS:", products.count())

embeddings = []

products_data = []


for product in products:

    try:

        print("PROCESSING:", product.name)

        if not product.image:
            continue

        image_path = product.image.path

        image = Image.open(
            image_path
        ).convert("RGB")

        image_input = preprocess(
            image
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            image_features = (
                model.encode_image(
                    image_input
                )
            )

        image_features /= (
            image_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        embedding = (
            image_features
            .cpu()
            .numpy()
            .astype("float32")
        )

        embeddings.append(
            embedding[0]
        )

        products_data.append({

            "id": product.id,

            "category":
                product.category,

            "subcategory":
                product.subcategory,

            "color":
                product.color
        })

        print("INDEXED:", product.name)

    except Exception as e:

        print("ERROR:", e)


print("CREATING FAISS INDEX")


if len(embeddings) == 0:

    print("NO EMBEDDINGS FOUND")

    exit()


dimension = len(
    embeddings[0]
)

index = faiss.IndexFlatIP(
    dimension
)

index.add(
    np.array(embeddings)
)

faiss.write_index(
    index,
    "faiss_index.bin"
)

print("FAISS SAVED")


with open(
    "products.pkl",
    "wb"
) as f:

    pickle.dump(
        products_data,
        f
    )

print("PRODUCT DATA SAVED")

print("DONE SUCCESSFULLY")
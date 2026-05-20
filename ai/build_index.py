import os
import django
import sys
import faiss
import pickle
import numpy as np

# Django setup
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

from products.models import Product
from ai.feature_extractor import extract_features


features = []
product_ids = []

products = Product.objects.all()

for product in products:
    try:
        image_path = product.image.path
        embedding = extract_features(image_path)

        if embedding is None:
            continue

        embedding = np.array(embedding, dtype="float32").flatten()

        norm = np.linalg.norm(embedding)
        if norm == 0:
            continue

        embedding = embedding / norm

        features.append(embedding)
        product_ids.append(product.id)

        print("Indexed:", product.name)

    except Exception as e:
        print("Error:", e)


features = np.array(features, dtype="float32")

if len(features) == 0:
    raise ValueError("No embeddings found!")

# ensure 2D
if len(features.shape) == 1:
    features = np.expand_dims(features, axis=0)

dimension = features.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(features)

faiss.write_index(index, "ai/faiss_index.bin")

with open("ai/product_ids.pkl", "wb") as f:
    pickle.dump(product_ids, f)

print("FAISS INDEX CREATED")
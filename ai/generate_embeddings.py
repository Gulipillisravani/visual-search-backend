import os
import sys
import django
import numpy as np
import faiss

# Add backend path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

# Django setup
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

from products.models import Product
from ai.clip_model import generate_embedding

# Get products
products = Product.objects.all()

embeddings = []
product_ids = []

for product in products:

    try:

        print(f"Processing: {product.name}")

        embedding = generate_embedding(product.image.path)

        if embedding is None:
            continue

        embeddings.append(embedding[0])
        product_ids.append(product.id)

    except Exception as e:
        print("ERROR:", e)

# Convert to numpy
embeddings = np.array(embeddings).astype("float32")

print("Embeddings Shape:", embeddings.shape)

# Dimension
dimension = embeddings.shape[1]

# IMPORTANT: Use COSINE similarity
index = faiss.IndexFlatIP(dimension)

# Add embeddings
index.add(embeddings)

# Save index
faiss.write_index(index, "ai/product_index.faiss")

# Save IDs
np.save("ai/product_ids.npy", np.array(product_ids))

print("✅ Embeddings Generated Successfully")
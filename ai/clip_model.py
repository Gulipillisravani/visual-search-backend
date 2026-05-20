from PIL import Image
import torch
import open_clip
import numpy as np

# ---------------------------------
# DEVICE
# ---------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using Device:", device)

# ---------------------------------
# LOAD SMALLER & FASTER MODEL
# ---------------------------------

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)

model.to(device)
model.eval()

# ---------------------------------
# GENERATE EMBEDDING
# ---------------------------------

def generate_embedding(image_path):

    try:

        # Open image
        image = Image.open(image_path).convert("RGB")

        # Resize image for speed
        image = image.resize((224, 224))

        # Preprocess
        image = preprocess(image).unsqueeze(0).to(device)

        # Generate features
        with torch.no_grad():
            features = model.encode_image(image)

        # Convert to numpy
        embedding = features.cpu().numpy().astype("float32")

        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    except Exception as e:

        print("Embedding Error:", e)

        return None
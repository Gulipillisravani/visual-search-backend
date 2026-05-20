import torch
import open_clip
import numpy as np

from PIL import Image


# LOAD CLIP MODEL
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)

tokenizer = open_clip.get_tokenizer(
    'ViT-B-32'
)


# EXTRACT AI FEATURES
def extract_features(image_path):

    image = preprocess(
        Image.open(image_path).convert("RGB")
    ).unsqueeze(0)

    with torch.no_grad():

        image_features = model.encode_image(
            image
        )

    # NORMALIZE FEATURES
    image_features /= image_features.norm(
        dim=-1,
        keepdim=True
    )

    return image_features.cpu().numpy()[0]


# DETECT DOMINANT COLOR
def detect_color(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize((100, 100))

    np_image = np.array(image)

    avg_color = np_image.mean(
        axis=(0, 1)
    )

    red, green, blue = avg_color

    if red > green and red > blue:
        return "Red"

    elif green > red and green > blue:
        return "Green"

    elif blue > red and blue > green:
        return "Blue"

    elif red > 180 and green > 180:
        return "Yellow"

    elif red < 80 and green < 80 and blue < 80:
        return "Black"

    else:
        return "White"


# DETECT PATTERN
def detect_pattern(image_path):

    # TEMP MOCK AI
    patterns = [
        "Floral",
        "Striped",
        "Checked",
        "Plain"
    ]

    return np.random.choice(patterns)


# DETECT STYLE
def detect_style(image_path):

    styles = [
        "Casual",
        "Party",
        "Sports",
        "Traditional"
    ]

    return np.random.choice(styles)


# DETECT CATEGORY
def detect_category(image_path):

    categories = [
        "Tshirt",
        "Saree",
        "Shoes",
        "Handbag",
        "Kurti"
    ]

    return np.random.choice(categories)


# FULL AI ANALYSIS
def analyze_image(image_path):

    features = extract_features(
        image_path
    )

    result = {

        "embedding": features,

        "color": detect_color(
            image_path
        ),

        "pattern": detect_pattern(
            image_path
        ),

        "style": detect_style(
            image_path
        ),

        "category": detect_category(
            image_path
        )

    }

    return result
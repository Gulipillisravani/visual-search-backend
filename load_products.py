import os
import random
import pandas as pd
import django

os.environ.setdefault(
    'DJANGO_SETTINGS_MODULE',
    'core.settings'
)

django.setup()

from products.models import Product


csv_path = "media/dataset/styles.csv"

image_folder = "media/dataset/images"


df = pd.read_csv(
    csv_path,
    on_bad_lines='skip'
)


for index, row in df.iterrows():

    try:

        product_id = row['id']

        image_name = f"{product_id}.jpg"

        image_path = os.path.join(
            image_folder,
            image_name
        )

        if not os.path.exists(image_path):

            continue

        Product.objects.create(

            product_id=product_id,

            name=str(
                row.get(
                    'productDisplayName',
                    'Unknown Product'
                )
            ),

            category=str(
                row.get(
                    'masterCategory',
                    'Fashion'
                )
            ),

            subcategory=str(
                row.get(
                    'subCategory',
                    'Clothing'
                )
            ),

            color=str(
                row.get(
                    'baseColour',
                    'Unknown'
                )
            ),

            gender=str(
                row.get(
                    'gender',
                    'Unisex'
                )
            ),

            season=str(
                row.get(
                    'season',
                    'All'
                )
            ),

            usage=str(
                row.get(
                    'usage',
                    'Casual'
                )
            ),

            price=random.randint(
                500,
                5000
            ),

            image=f"dataset/images/{image_name}"
        )

        print(f"Added: {image_name}")

    except Exception as e:

        print("Error:", e)


print("Products Loaded Successfully")
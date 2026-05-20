from django.db import models

class Product(models.Model):

    name = models.CharField(max_length=255)

    brand = models.CharField(max_length=100)

    category = models.CharField(max_length=100)

    color = models.CharField(max_length=100)

    price = models.DecimalField(
        max_digits=10,
        decimal_places=2
    )

    image = models.ImageField(
        upload_to='products/'
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return self.name

from django.db import models
from django.contrib.auth import get_user_model
from django.utils import timezone
# Create your models here.

User = get_user_model()

class PredictionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    input_data = models.JSONField() # Lưu trữ input dưới dạng JSON
    output_result = models.CharField(max_length=10) # >50K hoặc <=50K
    probability = models.FloatField() # Xác suất
    prediction_date = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f'{self.user.username} - {self.output_result} ({self.probability}%)'
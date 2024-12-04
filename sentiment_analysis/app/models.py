from django.db import models

class SentimentAnalysisModel(models.Model):
    text = models.TextField()
    sentiment = models.FloatField()
    created_at = models.DateTimeField()

    # def __str__(self):
    #     return f"Sentiment:"

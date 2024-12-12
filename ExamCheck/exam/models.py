from django.db import models

class AnswerSheet(models.Model):
    file = models.ImageField(upload_to='answersheets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Answer Sheet {self.id}"

class AnswerComparison(models.Model):
    answer_sheet = models.ForeignKey(AnswerSheet, on_delete=models.CASCADE)
    question_number = models.IntegerField()
    predicted_answer = models.CharField(max_length=1)
    correct_answer = models.CharField(max_length=1)
    confidence = models.FloatField()

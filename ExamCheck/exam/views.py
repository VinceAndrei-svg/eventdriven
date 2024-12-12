import os
from django.shortcuts import render, redirect
from .forms import AnswerSheetForm
from .models import AnswerSheet
from .utils import segment_bubbles, preprocess_image
from tensorflow.keras.models import load_model
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Load the trained model
MODEL_PATH = "C:/Users/vince/Desktop/polder1/dataset/Dataset/ExamCheck2.0.0.keras"
model = load_model(MODEL_PATH)

CLASS_NAMES = ['A', 'B', 'C', 'D']
ANSWER_KEY = ['A', 'C', 'B', 'D', 'A', 'B', 'C', 'D', 'A', 'C']  # Replace with your actual answer key

def predict_answers(image_path):
    """
    Segments the answer sheet into bubbles, predicts answers for each, and returns results.
    """
    # Segment the answer sheet into individual bubble images
    try:
        bubble_images = segment_bubbles(image_path)
    except Exception as e:
        logger.error(f"Error in segmenting bubbles: {e}")
        return []

    results = []
    for i, bubble_path in enumerate(bubble_images):
        try:
            img_array = preprocess_image(bubble_path)
            predictions = model.predict(img_array)

            print('TEST:')
            print(predictions)
            print(np.argmax(predictions))

            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = round(100 * np.max(predictions), 2)

            results.append({
                'question_number': i + 1,
                'predicted_answer': predicted_class,
                'correct_answer': ANSWER_KEY[i] if i < len(ANSWER_KEY) else None,
                'confidence': confidence,
            })
        except Exception as e:
            logger.error(f"Error in predicting answer for bubble {i+1}: {e}")

        # Remove the bubble image file after processing
        os.remove(bubble_path)

    return results

def upload_sheet(request):
    """
    Handles answer sheet uploads, processes predictions, and renders results.
    """
    if request.method == "POST":
        form = AnswerSheetForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Save uploaded answer sheet
                answer_sheet = form.save()

                # Predict answers
                results = predict_answers(answer_sheet.file.path)

                if not results:
                    return render(request, 'error.html', {'message': 'Failed to process the answer sheet.'})

                # Calculate the total correct answers
                correct_answers = sum(
                    r['predicted_answer'] == r['correct_answer'] for r in results
                )

                return render(request, 'result.html', {
                    'results': results,
                    'total_questions': len(results),
                    'correct_answers': correct_answers,
                })
            except Exception as e:
                logger.error(f"Error in processing uploaded sheet: {e}")
                return render(request, 'error.html', {'message': 'An error occurred during processing.'})
    else:
        form = AnswerSheetForm()
    return render(request, 'upload.html', {'form': form})

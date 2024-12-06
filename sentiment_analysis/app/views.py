from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
import requests
from bs4 import BeautifulSoup
import random
import threading
from .sentiment_model import SentimentAnalysisModel

# Global variables
sentiment_model = None
model_loading = False
model_ready = False

def check_model_status(request):
    """Endpoint to check if model is ready"""
    return JsonResponse({
        'status': 'ready' if model_ready else 'loading'
    })

def load_model_async():
    """Function to load model in background"""
    global sentiment_model, model_loading, model_ready
    try:
        sentiment_model = SentimentAnalysisModel(verbose=True)
        if sentiment_model and sentiment_model.is_trained:
            print("Model loaded successfully and is trained")
            model_ready = True
        else:
            print("Warning: Model loaded but not trained")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
    model_loading = False

def home(request):
    return render(request, "home.html")

def scrape_reviews(request):
    global sentiment_model, model_loading, model_ready
    reviews = []
    analysis = None
    context = {}

    # Start model loading if it hasn't been started
    if sentiment_model is None and not model_loading:
        print("Starting model load")
        model_loading = True
        model_ready = False
        thread = threading.Thread(target=load_model_async)
        thread.start()
    
    if request.method == 'POST':
        url = request.POST.get('amazon_url')
        try:
            # Headers to mimic a browser request
            # headers = {
            #     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            # }
            
            response = requests.get(url) # , headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find review elements
            review_elements = soup.find_all('div', {'data-hook': 'review'})
            for review in review_elements:
                review_text = review.find('span', {'data-hook': 'review-body'})
                if review_text:
                    reviews.append(review_text.text.strip())

            star_rating_element = soup.find('span', {'data-hook': 'rating-out-of-text'})
            avg_stars = star_rating_element.text.split(' ')[0] if star_rating_element else "0.0"

            # Analyze sentiment if model is ready
            if model_ready and sentiment_model and sentiment_model.is_trained:
                print("Using trained model for analysis")
                positive_count = 0
                for review in reviews:
                    score = sentiment_model.analyze_text(review)
                    if score > 0.5:  # threshold for positive sentiment
                        positive_count += 1
                
                positive_percent = (positive_count / len(reviews) * 100) if reviews else 0
                analysis = {
                    'positive_percent': round(positive_percent, 1),
                    'average_rating': avg_stars,
                    'avg_percent': round((float(avg_stars)/5)*100, 1)
                }
            else:
                print("Using random analysis (model not ready)")
                analysis = {
                    'positive_percent': random.randint(50, 98),
                    'average_rating': avg_stars,
                    'avg_percent': round((float(avg_stars)/5)*100, 1)
                }
            print("Analysis:", analysis)
                    
        except Exception as e:
            reviews = [f"Error: {str(e)}"]
            print("Error:", str(e))

        context = {
            'reviews': reviews,
            'analysis': analysis,
            'model_ready': model_ready
        }
    
    return render(request, 'test.html', context)
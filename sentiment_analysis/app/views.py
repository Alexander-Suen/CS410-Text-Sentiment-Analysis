from django.shortcuts import render
from django.http import HttpResponse
import requests
from bs4 import BeautifulSoup
import re
import random

# Create your views here.
def home(request):
    return render(request, "home.html")

def scrape_reviews(request):
    reviews = []
    analysis = None
    context = {}
    
    if request.method == 'POST':
        url = request.POST.get('amazon_url')
        try:
            # Headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find review elements
            review_elements = soup.find_all('div', {'data-hook': 'review'})
            # stars = soup.find('span', {'data-hook': 'rating-out-of-text'})
            # avg_stars = stars.text
            # print(avg_stars)
            for review in review_elements:
                review_text = review.find('span', {'data-hook': 'review-body'})
                if review_text:
                    reviews.append(review_text.text.strip())

            star_rating_element = soup.find('span', {'data-hook': 'rating-out-of-text'})  # Replace with actual class name on the page

            avg_stars = star_rating_element.text.split(' ')[0]
            
            # Generate random analysis data
            analysis = {
                'positive_percent': random.randint(50, 98),
                'average_rating': avg_stars,
                'avg_percent': (float(avg_stars)/5)*100
                # 'average_rating': round(random.uniform(3.5, 4.9), 1)
            }
            print("Analysis:", analysis)  # Debug print
                    
        except Exception as e:
            reviews = [f"Error: {str(e)}"]
            print("Error:", str(e))  # Debug print

        context = {
            'reviews': reviews,
            'analysis': analysis
        }
    
    
    print("Context:", context)  # Debug print
    return render(request, 'test.html', context)


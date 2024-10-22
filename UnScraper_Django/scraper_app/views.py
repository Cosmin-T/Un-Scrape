# scraper_app/views.py
from django.shortcuts import render
from django.views.decorators.cache import never_cache
import requests

@never_cache  # Prevents caching of the view
def scrape_website(request):
    # Initialize context
    context = {
        'rows': None,
        'error': None,
        'show_results': False
    }

    # Only process if it's a real POST request with form data
    if request.method == 'POST' and 'url' in request.POST:
        url = request.POST.get('url')
        groq_api_key = request.POST.get('groq_api_key')
        fields = request.POST.get('fields').split(',')

        data = {
            "url": url,
            "groq_api_key": groq_api_key,
            "fields": [{"name": field.strip()} for field in fields]
        }

        try:
            response = requests.post('http://localhost:8090/scrape', json=data)
            response.raise_for_status()

            result = response.json()
            if 'listings' in result:
                context.update({
                    'rows': result['listings'],
                    'show_results': True
                })
            else:
                context['error'] = 'No data found in response'

        except requests.exceptions.RequestException as e:
            context['error'] = f'Failed to scrape the website: {str(e)}'

    return render(request, 'index.html', context)
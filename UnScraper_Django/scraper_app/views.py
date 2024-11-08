# scraper_app/views.py
from django.shortcuts import render
from django.views.decorators.cache import never_cache
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect
from django.core.validators import URLValidator
from django.core.exceptions import ValidationError
from bs4 import BeautifulSoup
import html2text
from groq import Groq
import json
import re
import asyncio
import concurrent.futures
from playwright.async_api import async_playwright
import random
from typing import List
import logging
from django.http import HttpResponse
import csv
from dotenv import load_dotenv, dotenv_values
import os

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
# print(f'Dotenv path is: {dotenv_path}')
load_dotenv(dotenv_path)

env_variables = dotenv_values(dotenv_path)

GROQ_API_KEY = env_variables.get('GROQ_API_KEY')
# print(f'Groq Key is: {GROQ_API_KEY}')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User agents list (truncated for brevity - you can keep the full list from api.py)
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
]

class ScraperError(Exception):
    """Custom exception for scraper-related errors"""
    pass

def groq_connection(api_key: str) -> Groq:
    """Initialize Groq client with error handling"""
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        logger.error(f"Error creating Groq client: {e}")
        raise ScraperError(f"Failed to initialize Groq API: {str(e)}")

async def fetch_and_clean_html(url: str) -> str:
    """Fetch and clean HTML content using Playwright"""
    try:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={'width': 1920, 'height': 1080},
                java_script_enabled=True,
                bypass_csp=True,
                ignore_https_errors=True,
            )

            context.set_default_timeout(15000)
            page = await context.new_page()

            # Block unnecessary resources
            await page.route("**/*.{png,jpg,jpeg,gif,svg}", lambda route: route.abort())
            await page.route("**/*analytics*.js", lambda route: route.abort())
            await page.route("**/*tracking*.js", lambda route: route.abort())
            await page.route("**/*advertisement*.js", lambda route: route.abort())

            logger.info(f"Navigating to {url}")
            await page.goto(url, wait_until='domcontentloaded')

            # Scroll to trigger lazy loading
            for _ in range(30):
                await page.mouse.wheel(0, 2000)
                await page.wait_for_timeout(150)

            try:
                await page.wait_for_load_state('networkidle', timeout=10000)
            except Exception:
                pass

            html_content = await page.content()

            # Clean HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
                element.decompose()

            # Convert to markdown/text
            markdown_converter = html2text.HTML2Text()
            markdown_converter.ignore_links = False
            markdown_converter.ignore_images = True
            markdown_converter.ignore_emphasis = False
            markdown_converter.ignore_tables = False
            markdown_converter.body_width = 0

            text_content = markdown_converter.handle(str(soup))

            # Clean content
            space_pattern = re.compile(r'\s+')
            url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

            cleaned_content = space_pattern.sub(' ', text_content)
            cleaned_content = url_pattern.sub('', cleaned_content)

            return cleaned_content.strip()

    except Exception as e:
        logger.error(f"Error fetching HTML: {e}")
        raise ScraperError(f"Failed to fetch webpage content: {str(e)}")

def clean_json_string(json_str: str) -> str:
    """Clean and validate JSON string"""
    try:
        # Find the start and end of the JSON object
        start = json_str.find('{')
        end = json_str.rfind('}') + 1

        # Check if valid JSON structure is found
        if start >= 0 and end > 0:
            # Extract the JSON object
            json_str = json_str[start:end]

            # Ensure the JSON string ends with ']}' for proper listing format
            if not json_str.endswith(']}'):
                # Remove trailing comma if present and add closing brackets
                json_str = json_str.rstrip(',') + ']}'

            return json_str
    except Exception as e:
        # Log any errors encountered during the cleaning process
        logger.error(f"Failed to clean JSON: {e}")

    # Return an empty listings object if cleaning fails
    return '{"listings": []}'

def process_chunk(client: Groq, sys_message: str, chunk: str, fields: List[str]) -> List[dict]:
    """ Process a chunk of text using Groq API" """
    # List of available LLM models
    llms = [
        'llama-3.2-90b-text-preview',
        'llama-3.2-90b-vision-preview',
        'llama-3.1-70b-versatile',
        'llama3-70b-8192',
        'llama3-groq-70b-8192-tool-use-preview'
    ]

    # Initialize model index and retry counter
    current_model_index = 0
    retry_count = 0

    # Retry loop with a maximum of 3 attempts
    while retry_count < 3:
        try:
            # Select current LLM model
            llm = llms[current_model_index]
            logger.info(f"Processing chunk with model {llm}")

            # Make API call to Groq
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_message},
                    {"role": "user", "content": (
                        f'Extract these fields from the text: {", ".join(fields)}.\n'
                        f'Return as JSON with format {{"listings": [{{fields}}]}}.\n'
                        f'Content:\n{chunk}'
                    )}
                ],
                model=llm,
                temperature=0.1
            )

            # Extract completion from response
            completion = response.choices[0].message.content
            logger.info(f"Raw LLM Response (truncated): {completion[:200]}...")

            try:
                # Clean and parse JSON response
                cleaned_completion = clean_json_string(completion)
                parsed_chunk = json.loads(cleaned_completion)
                if 'listings' not in parsed_chunk:
                    raise ValueError("Missing 'listings' key")
                return parsed_chunk['listings']
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback parsing if JSON is invalid
                logger.error(f"Error parsing response: {e}")
                listings = re.findall(r'\{[^{}]*\}', completion)
                if listings:
                    return [json.loads(clean_json_string(listing)) for listing in listings]
                retry_count += 1
                continue

        except Exception as e:
            # Handle errors, including rate limiting
            logger.error(f"Error in process_chunk: {e}")
            if "rate_limit" in str(e).lower():
                current_model_index = (current_model_index + 1) % len(llms)
                logger.info(f"Rate limit hit, switching to model {llms[current_model_index]}")
            else:
                retry_count += 1

    # Return empty list if all retries fail
    return []

@csrf_protect
@never_cache
async def scrape_website(request):
    """Main view function for website scraping"""
    context = {
        'rows': None,
        'error': None,
        'show_results': False,
        'default_api_key': GROQ_API_KEY
    }

    if request.method == 'POST':
        try:
            # Extract and validate inputs
            url = request.POST.get('url')
            groq_api_key = request.POST.get('groq_api_key')
            fields = [field.strip() for field in request.POST.get('fields', '').split(',') if field.strip()]

            # Validate URL
            url_validator = URLValidator()
            try:
                url_validator(url)
            except ValidationError:
                raise ScraperError("Please provide a valid URL starting with http:// or https://")

            # Validate fields
            if not fields:
                raise ScraperError("Please specify at least one field to extract")

            # Initialize Groq client
            client = groq_connection(groq_api_key)

            # Fetch and process content
            content = await fetch_and_clean_html(url)
            chunk_size = 10000
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

            sys_message = """
                You are a data extraction expert. Extract structured information from the given text.
                Return ONLY a valid JSON object containing the requested fields.
                The response MUST be in this exact format, with no additional text:
                {"listings": [{"field1": "value1", "field2": "value2"}, ...]}
                Each listing must include all requested fields. Use an empty string if a field is not found.
                Ensure all quotes are double quotes and there are no trailing commas.
                """

            # Process chunks concurrently
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                tasks = [
                    loop.run_in_executor(pool, process_chunk, client, sys_message, chunk, fields)
                    for chunk in chunks
                ]
                results = await asyncio.gather(*tasks)

            # Combine results
            all_listings = []
            for result in results:
                if isinstance(result, list):
                    all_listings.extend(result)

            if not all_listings:
                raise ScraperError("Could not extract any data with the specified fields")

            context.update({
                'rows': all_listings,
                'show_results': True
            })

        except ScraperError as e:
            context['error'] = str(e)
            logger.error(f"Scraping error: {e}")
        except Exception as e:
            context['error'] = f"An unexpected error occurred: {str(e)}"
            logger.error(f"Unexpected error: {e}", exc_info=True)

    return render(request, 'index.html', context)

@csrf_protect
@never_cache
def download_csv(request):
    """
    Handle CSV download requests for scraped data.
    """
    if request.method == 'POST':
        try:
            # Parse the JSON data from the request body
            data = json.loads(request.body)
            rows = data.get('rows', [])

            # Set up the HTTP response for CSV download
            response = HttpResponse(content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="scraping_results.csv"'

            if rows:
                # Create a CSV writer and write the data
                writer = csv.DictWriter(response, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            return response
        except Exception as e:
            # Return error response if an exception occurs
            return JsonResponse({'error': str(e)}, status=400)
    # Return error for non-POST requests
    return JsonResponse({'error': 'Invalid request'}, status=400)

@csrf_protect
@never_cache
def download_json(request):
    """
    Handle JSON download requests for scraped data.
    """
    if request.method == 'POST':
        try:
            # Parse the JSON data from the request body
            data = json.loads(request.body)
            rows = data.get('rows', [])

            # Set up the HTTP response for JSON download
            response = HttpResponse(content_type='application/json')
            response['Content-Disposition'] = 'attachment; filename="scraping_results.json"'

            # Write the JSON data to the response
            json.dump(rows, response, indent=2)

            return response
        except Exception as e:
            # Return error response if an exception occurs
            return JsonResponse({'error': str(e)}, status=400)
    # Return error for non-POST requests
    return JsonResponse({'error': 'Invalid request'}, status=400)
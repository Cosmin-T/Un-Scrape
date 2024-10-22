from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import json
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import html2text
from groq import Groq
import random
import time
import re
import asyncio
import concurrent.futures

app = FastAPI()

class ListingField(BaseModel):
    """
    Represents a field that should be extracted from the page.
    """

    name: str

class RequestData(BaseModel):
    """
    Represents the data sent in the request body to the /scrape endpoint.
    """
    url: str
    groq_api_key: str
    fields: List[ListingField]

def groq_connection(api_key):
    """
    Establishes a connection with the Groq API using the given API key.

    Returns the Groq client object if the connection is successful, otherwise None.
    """
    # Try to create the Groq client
    try:
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        # If an error occurs, print the error message and return None
        print(f"Error creating Groq client: {e}")
        return None

async def fetch_and_clean_html(url):
    """
    Fetches a webpage using Playwright, scrolls to load all dynamic content,
    and then cleans the HTML to extract readable text.

    Args:
        url (str): The URL of the webpage to fetch and clean.

    Returns:
        str: The cleaned text content of the webpage in markdown format.
    """

    # Launch Playwright and create a new browser instance
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()

        # Navigate to the specified URL
        print(f"Navigating to {url}")
        await page.goto(url)

        # Wait until the network is idle to ensure all elements are loaded
        await page.wait_for_load_state('networkidle')

        # Scroll to the bottom of the page to load dynamic content
        last_height = await page.evaluate('document.body.scrollHeight')
        while True:
            await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            # Wait for new content to load
            await page.wait_for_timeout(2000)
            new_height = await page.evaluate('document.body.scrollHeight')
            if new_height == last_height:
                break
            last_height = new_height

        # Retrieve the full HTML content of the loaded page
        html_content = await page.content()
        await browser.close()

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove script and style elements to avoid non-content data
        for element in soup(['script', 'style']):
            element.decompose()

        # Convert HTML to markdown using html2text
        markdown_converter = html2text.HTML2Text()
        markdown_converter.ignore_links = True  # Ignore link conversion
        markdown_converter.ignore_images = True  # Ignore image conversion
        markdown_converter.ignore_emphasis = True  # Ignore emphasis styles
        markdown_converter.body_width = 0  # Disable line wrapping

        # Convert the cleaned HTML to markdown format
        text_content = markdown_converter.handle(str(soup))
        # Remove excessive newlines and trim whitespace
        cleaned_content = re.sub(r'\n+', '\n', text_content).strip()
        return cleaned_content

def clean_json_response(response):
    """
    Clean a JSON response by removing any leading or trailing text.

    This is necessary because the JSON responses are embedded in Markdown code blocks
    and may contain other text before or after the JSON content.

    :param str response: JSON response to clean
    :return str: Cleaned JSON response
    """

    # Remove any leading or trailing text
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)

    # Remove any text before the first {
    start_idx = response.find('{')
    if start_idx != -1:
        response = response[start_idx:]

    # Remove any text after the last }
    end_idx = response.rfind('}')
    if end_idx != -1:
        response = response[:end_idx+1]

    return response.strip()

def process_chunk(client, sys_message, chunk, fields):
    """
    Process a chunk of text by extracting the given fields using the Groq API.

    The Groq API is used to extract the given fields from the text chunk.
    The response is cleaned and converted to a JSON object.
    If the response is invalid, the function will retry with a different model
    up to 3 times.

    :param client: The Groq client object
    :param sys_message: The system message to send to the LLM
    :param chunk: The text chunk to process
    :param fields: The fields to extract
    :return: A list of dictionaries with the extracted fields
    """
    # Choose a random model from the available models
    llms = ['llama3-70b-8192', 'llama-3.1-70b-versatile']
    llm = random.choice(llms)

    # Initialize the retry count and delay
    retry_count = 0
    delay = 1

    # Loop until we have successfully processed the chunk or we have retried too many times
    while retry_count < 3:
        try:
            print(f"Processing chunk with model {llm}")

            # Send the request to the LLM and get the response
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_message},
                    {"role": "user", "content": (
                        # Create the input to the LLM
                        f'Extract these fields from the text: {", ".join(fields)}.\n'
                        f'Return as JSON with format {{"listings": [{{fields}}]}}.\n'
                        f'Content:\n{chunk}'
                    )}
                ],
                # Send the request to the LLM
                model=llm,
                temperature=0.1
            )

            # Get the completion from the response
            completion = response.choices[0].message.content
            print(f"Raw LLM Response: {completion[:200]}...")

            # Clean the JSON response
            cleaned_json = clean_json_response(completion)
            print(f"Cleaned JSON: {cleaned_json[:200]}...")

            # Parse the cleaned JSON response
            try:
                parsed_chunk = json.loads(cleaned_json)
                if 'listings' not in parsed_chunk:
                    raise ValueError("Missing 'listings' key")
                return parsed_chunk['listings']
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing response: {e}")
                retry_count += 1
                delay *= 2
                time.sleep(delay + random.uniform(0, 1))
                continue

        except Exception as e:
            print(f"Error in process_chunk: {e}")
            retry_count += 1
            delay *= 2
            time.sleep(delay + random.uniform(0, 1))

    return []

@app.post("/scrape")
async def scrape_website(data: RequestData):
    """
    Scrape a website and extract structured information.

    This endpoint takes a URL, Groq API key, and list of fields to extract.
    It fetches the page content, chunks it into smaller pieces, and uses
    the Groq AI to extract the requested fields from each chunk.
    The responses are cleaned and combined into a single JSON object.

    Returns a JSON object with a single key "listings", which contains a list
    of dictionaries, each representing a single listing with the extracted fields.
    """
    try:
        # Create a Groq client object using the given API key
        client = groq_connection(data.groq_api_key)

        # Check if the client was successfully created
        if not client:
            raise HTTPException(status_code=500, detail="Failed to connect to Groq")

        # Get the list of fields to extract
        fields = [field.name for field in data.fields]

        # Fetch the page content
        content = await fetch_and_clean_html(data.url)

        # Split the content into chunks of a fixed size
        chunk_size = 10000
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

        # Create the system message to be sent to the AI
        sys_message = f"""
        You are a data extraction expert. Extract structured information from the given text.
        Only return a valid JSON object containing the requested fields.
        Format the response exactly like this, with no additional text or formatting:
        {{"listings": [{{"field1": "value1", "field2": "value2"}}]}}
        Include all requested fields in each listing, using empty string if not found.
        """

        # Process each chunk in parallel using a thread pool
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            tasks = [
                loop.run_in_executor(pool, process_chunk, client, sys_message, chunk, fields)
                for chunk in chunks
            ]
            results = await asyncio.gather(*tasks)

        # Combine the results from each chunk
        all_listings = []
        for result in results:
            if isinstance(result, list):
                all_listings.extend(result)

        # Return the final JSON object
        return {"listings": all_listings}

    except Exception as e:
        print(f"Error in scrape_website: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
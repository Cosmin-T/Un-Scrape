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
    name: str

class RequestData(BaseModel):
    url: str
    groq_api_key: str
    fields: List[ListingField]

def groq_connection(api_key):
    try:
        client = Groq(api_key=api_key)
        return client
    except Exception as e:
        print(f"Error creating Groq client: {e}")
        return None

async def fetch_and_clean_html(url):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080}
        )
        page = await context.new_page()

        print(f"Navigating to {url}")
        await page.goto(url)

        # Wait for content to load
        await page.wait_for_load_state('networkidle')

        # Scroll to load all dynamic content
        last_height = await page.evaluate('document.body.scrollHeight')
        while True:
            await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            await page.wait_for_timeout(2000)
            new_height = await page.evaluate('document.body.scrollHeight')
            if new_height == last_height:
                break
            last_height = new_height

        html_content = await page.content()
        await browser.close()

        # Clean HTML content
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup(['script', 'style']):
            element.decompose()

        markdown_converter = html2text.HTML2Text()
        markdown_converter.ignore_links = True
        markdown_converter.ignore_images = True
        markdown_converter.ignore_emphasis = True
        markdown_converter.body_width = 0

        text_content = markdown_converter.handle(str(soup))
        cleaned_content = re.sub(r'\n+', '\n', text_content).strip()
        return cleaned_content

def clean_json_response(response):
    # Remove any markdown code block markers
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
    llms = ['llama3-70b-8192', 'llama-3.1-70b-versatile']
    llm = random.choice(llms)
    retry_count = 0
    delay = 1

    while retry_count < 3:
        try:
            print(f"Processing chunk with model {llm}")
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

            completion = response.choices[0].message.content
            print(f"Raw LLM Response: {completion[:200]}...")

            # Clean the JSON response
            cleaned_json = clean_json_response(completion)
            print(f"Cleaned JSON: {cleaned_json[:200]}...")

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
    try:
        print("Starting scrape request")
        client = groq_connection(data.groq_api_key)

        if not client:
            raise HTTPException(status_code=500, detail="Failed to connect to Groq")

        fields = [field.name for field in data.fields]
        print(f"Requested fields: {fields}")

        content = await fetch_and_clean_html(data.url)
        print("Fetched page content")

        chunk_size = 10000
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

        sys_message = f"""
        You are a data extraction expert. Extract structured information from the given text.
        Only return a valid JSON object containing the requested fields.
        Format the response exactly like this, with no additional text or formatting:
        {{"listings": [{{"field1": "value1", "field2": "value2"}}]}}
        Include all requested fields in each listing, using empty string if not found.
        """

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            tasks = [
                loop.run_in_executor(pool, process_chunk, client, sys_message, chunk, fields)
                for chunk in chunks
            ]
            results = await asyncio.gather(*tasks)

        all_listings = []
        for result in results:
            if isinstance(result, list):
                all_listings.extend(result)

        print(f"Processed {len(all_listings)} listings")
        return {"listings": all_listings}

    except Exception as e:
        print(f"Error in scrape_website: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
# scraper.py

import os
import random
import time
import re
import json
from datetime import datetime
from playwright.sync_api import sync_playwright, Locator
import pandas as pd
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, create_model
import html2text
import tiktoken
import concurrent.futures
import sys
import streamlit as st
from logic.util import *
from logic.groq_conn import *

def connect_to_llm():
    """
    Connect to the LLM and retrieve the list of user agents.

    Returns:
        List[str]: List of user agents

    Raises:
        Exception: If the connection to the LLM fails
    """
    # The user agents are globally defined in the util.py file so we just return it here
    try:
        return USER_AGENTS
    # If the connection to the LLM fails, print an error message
    except Exception as e:
        print(f"Failed to connect to LLM: {e}")
        # Return None to indicate failure
        return None
def setup_playwright(url=None):
    """
    Sets up the Playwright browser context and page.

    Connects to Groq and LLM, and then sets up a headless Chromium browser with a random user agent.
    Returns a tuple of (playwright, browser, context, page) if successful, otherwise None.

    Raises:
        Exception: If an error occurs while connecting to Groq or LLM, or setting up the browser
    """
    try:
        # Connect to Groq and LLM
        client = groq_connection(st.session_state['groq_key'])
        if client is None:
            print("Could not create Groq client")
            return

        llm_user_agents = connect_to_llm()
        if llm_user_agents is None:
            print("Could not connect to LLM")
            return

        # Set up a headless Chromium browser with a random user agent
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=random.choice(llm_user_agents),
            viewport={'width': 1920, 'height': 1080}
        )
        page = context.new_page()
        if url is not None:
            page.goto(url)
        return playwright, browser, context, page
    except Exception as e:
        # If an error occurs, print an error message and return None
        print(f"An error occurred: {e}")
        return

def click_accept_cookies(page):
    """
    Clicks the "Accept Cookies" button on a webpage, if it exists.

    This function is a workaround for webpages that require the user to accept cookies
    before showing the content. It will try to find a button with text matching any of the
    following: accept, agree, allow, authorize, confirm, continue, ok, okay, alright,
    I agree, got it, proceed, yes, yup, I consent, accept all, accept terms,
    acknowledge, confirm understanding, next, done, submit, send, approve, validate.

    If the button is found and visible, it will be clicked. If the button is not found,
    a message will be printed to the console stating that it was not found. If an error
    occurs while trying to find the button, the error will be printed to the console.

    Args:
        page (playwright.Page): The page object to search for the button on.

    Returns:
        None
    """

    # Try to find the button using a list of possible text variations and tag names (button, a, div)
    try:
        accept_text_variations = [
            "accept", "agree", "allow", "authorize", "confirm",
            "continue", "ok", "okay", "alright", "I agree", "got it",
            "proceed", "yes", "yup", "I consent", "accept all", "accept terms",
            "acknowledge", "confirm understanding", "next", "done",
            "submit", "send", "approve", "validate"
        ]

        for tag in ["button", "a", "div"]:
            for text in accept_text_variations:
                try:
                    # Construct the selector using the tag name and text variation
                    selector = f"{tag}:has-text('{text}')"
                    # Find the first element that matches the selector
                    element = page.locator(selector).first
                    # Check if the element is visible
                    if element.is_visible():
                        # Click the element if it is visible
                        element.click()
                        # Print a success message
                        print(f"Clicked the '{text}' button.")
                        # Return so that the function exits
                        return
                # If an error occurs, continue to the next iteration
                except:
                    continue

        # If the loop completes without finding the button, print a message
        print("No 'Accept Cookies' button found.")

    # If an exception occurs, print the error message
    except Exception as e:
        print(f"Error finding 'Accept Cookies' button: {e}")

def fetch_html_playwright(url):
    """
    Fetches the HTML of a webpage using playwright.

    Args:
        url (str): The URL of the webpage to fetch.

    Returns:
        str: The HTML of the webpage.

    Notes:
        This function uses playwright to load the webpage and extracts the HTML
        content. It also handles clicking the "Accept Cookies" button if it is
        present.
    """
    # Set up playwright
    playwright, browser, context, page = setup_playwright(url)
    try:
        # Navigate to the webpage
        page.goto(url)
        # Wait some time for the page to fully load
        time.sleep(random.uniform(1, 3))
        # Click the "Accept Cookies" button if it is present
        click_accept_cookies(page)
        # Scroll to the bottom of the page to ensure all content is loaded
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        # Wait some time for the page to fully load after scrolling
        time.sleep(random.uniform(1, 2))
        # Scroll to the bottom of the page again to ensure all content is loaded
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        # Wait some time for the page to fully load after scrolling
        time.sleep(random.uniform(0.5, 1))
        # Get the HTML content of the page
        html = page.content()
        # Return the HTML content
        return html
    # Close the browser and stop playwright
    finally:
        browser.close()
        playwright.stop()

def clean_html(html_content):
    """
    Removes all header and footer tags from the given HTML content.

    Args:
        html_content (str): The HTML content to clean.

    Returns:
        str: The cleaned HTML content.
    """
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all header and footer tags and remove them from the parsed HTML
    for element in soup.find_all(['header', 'footer']):
        element.decompose()

    # Return the cleaned HTML content as a string
    return str(soup)

def html_to_markdown_with_readability(html_content):
    """
    Converts the given HTML content to Markdown format, using the
    `html2text` library. The HTML content is first cleaned of header and
    footer tags.

    Args:
        html_content (str): The HTML content to convert.

    Returns:
        str: The Markdown content.
    """
    # Clean the input HTML by removing all header and footer tags
    cleaned_html = clean_html(html_content)
    # Create an html2text converter object
    markdown_converter = html2text.HTML2Text()
    # Do not ignore links in the conversion
    markdown_converter.ignore_links = False
    # Convert the cleaned HTML to Markdown
    markdown_content = markdown_converter.handle(cleaned_html)

    # Return the Markdown content
    return markdown_content

def save_raw_data(raw_data, timestamp, output_folder='output'):
    """
    Saves the given raw data to a file in the specified output folder.

    The file name is in the format "rawData_<timestamp>.md".

    Args:
        raw_data (str): The raw data to save.
        timestamp (str): The timestamp to use in the file name.
        output_folder (str): The folder where the file will be saved.
            Defaults to 'output'.

    Returns:
        str: The path to the saved file.
    """
    # Create the output folder if it does not already exist
    os.makedirs(output_folder, exist_ok=True)

    # Create the full path to the output file
    raw_output_path = os.path.join(output_folder, f'rawData_{timestamp}.md')

    # Write the raw data to the output file
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        f.write(raw_data)

    # Print out the path to the saved file
    print(f"Raw data saved to {raw_output_path}")

    # Return the path to the saved file
    return raw_output_path

def remove_urls_from_file(file_path):
    """
    Removes URLs from a given Markdown file and saves a cleaned version.

    The cleaned file is saved with the same name as the original file, but with
    "_cleaned" appended to the base name.

    Args:
        file_path (str): The path to the Markdown file to clean.

    Returns:
        str: The cleaned Markdown content.
    """

    # The pattern to match URLs is a regular expression that will match most
    # common URLs. It does not match URLs with whitespace in them, as these are not typically valid URLs.
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Split the file path into its base name and extension
    base, ext = os.path.splitext(file_path)
    # Construct the name of the new file to be saved
    new_file_path = f"{base}_cleaned{ext}"

    # Read the contents of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        markdown_content = file.read()

    # Remove all URLs from the file contents
    cleaned_content = re.sub(url_pattern, '', markdown_content)

    # Save the cleaned contents to the new file
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_content)
    print(f"Cleaned file saved as: {new_file_path}")
    # Return the cleaned contents
    return cleaned_content

def create_dynamic_listing_model(field_names):
    """
    Creates a dynamic Pydantic model based on a list of field names.

    Args:
        field_names (List[str]): A list of field names to include in the model.

    Returns:
        Type[BaseModel]: A dynamic Pydantic model with string fields for each of the given field names.
    """
    # Create a dictionary of field definitions, where each field is a string field with no default value. The `...` syntax indicates that the field has no default value.
    field_definitions = {field: (str, ...) for field in field_names}
    # Use the create_model function to create the dynamic model with the given field definitions.
    return create_model('DynamicListingModel', **field_definitions)

def create_listings_container_model(listing_model):
    """
    Creates a dynamic Pydantic model that contains a field called "listings"
    which is a list of the given listing model.

    Args:
        listing_model: The Pydantic model to use for the listings field.

    Returns:
        A dynamic Pydantic model with a listings field.
    """

    # Create a dictionary of field definitions, where each field is a string field with no default value. The `...` syntax indicates that the field has no default value.
    field_definitions = {'listings': (list[listing_model], ...)}
    # Use the create_model function to create the dynamic model with the given field definitions.
    return create_model('DynamicListingsContainer', **field_definitions)

def trim_to_token_limit(text, model, max_tokens=120000):
    """
    Trim a given text to a given token limit using the tiktoken library.

    Args:
        text (str): The text to trim.
        model (str): The tiktoken model to use for tokenization.
        max_tokens (int, optional): The maximum number of tokens to allow in the text. Defaults to 120000.

    Returns:
        str: The trimmed text, or the original text if it was within the token limit.
    """
    try:
        # Get the tiktoken encoder for the given model
        encoder = tiktoken.encoding_for_model(model)
    except Exception as e:
        print(f"Error getting encoder for model {model}: {e}")
        return text

    # Encode the text into tokens
    tokens = encoder.encode(text)

    # Check if the number of tokens exceeds the maximum allowed
    if len(tokens) > max_tokens:
        # Trim the tokens to the maximum allowed
        trimmed_tokens = tokens[:max_tokens]
        # Decode the trimmed tokens back into text
        trimmed_text = encoder.decode(trimmed_tokens)
        # Return the trimmed text
        return trimmed_text
    # Return the original text if it was within the token limit
    return text
def generate_system_message(listing_model):
    """
    Generates a system message for the AI that will process the given text, which describes the expected output format.

    The system message includes a description of the task, and the expected JSON schema of the output.

    Args:
        listing_model: The Pydantic model that describes the JSON schema of the output.

    Returns:
        str: The system message that will be sent to the AI.
    """
    # Get the JSON schema for the given model
    schema_info = listing_model.model_json_schema()

    # Create a list of field descriptions, which are just the field names and types
    field_descriptions = []
    for field_name, field_info in schema_info["properties"].items():
        # Use the type field from the schema info, but if it's a list, use the first item in the list
        field_type = (
            field_info["type"][0] if isinstance(field_info["type"], list) else field_info["type"]
        )
        field_descriptions.append(f'"{field_name}": "{field_type}"')

    # Join the field descriptions into a single string with commas and newlines
    schema_structure = ",\n".join(field_descriptions)

    # Create the system message with the task description and the expected output schema
    system_message = f"""
    You are an intelligent text extraction and conversion assistant. Your task is to extract structured information
    from the given text and convert it into a pure JSON format. The JSON should contain only the structured data extracted from the text,
    with no additional commentary, explanations, or extraneous information.
    You could encounter cases where you can't find the data of the fields you have to extract or the data will be in a foreign language.
    Please process the following text and provide the output in pure JSON format with no words before or after the JSON:
    Please ensure the output strictly follows this schema:

    {{
        "listings": [
            {{
                {schema_structure}
            }}
        ]
    }}
    """

    return system_message

def process_chunk(client, sys_message, chunk):
    """
    Process a chunk of text and return the prompt tokens, completion tokens, and parsed chunk.

    Args:
        client (openai.ChatCompletion): The OpenAI chat completion client.
        sys_message (str): The system message to send to the AI.
        chunk (str): The chunk of text to process.

    Returns:
        tuple: A tuple containing the prompt tokens, completion tokens, and parsed chunk.
    """
    # Set the model to use for the AI at a random choice from the list
    llms = ['llama3-70b-8192','llama3-8b-8192',
            'llama-3.1-70b-versatile','llama-3.1-8b-instant','llama3-groq-70b-8192-tool-use-preview',
            'llama-3.2-90b-text-preview','llama-3.2-11b-text-preview']
    llm = random.choice(llms)

    # The retry count starts at 0, and we will retry up to 3 times if an error occurs.
    retry_count = 0

    # The delay starts at 1 second, and will increase exponentially (up to a maximum of 30 seconds) if the same error occurs multiple times.
    delay = 1

    # The rate limit retry count starts at 0, and will increment each time we hit the rate limit.
    # If we hit the rate limit too many times, we will give up and return an empty list.
    rate_limit_retries = 0
    max_rate_limit_retries = 5

    # The loop will continue until we have either successfully processed the chunk, or we have retried too many times.
    while retry_count < 3:
        try:
            # Print a message to the console indicating that we are sending a request to the AI.
            print(f"Sending request with chunk: {chunk[:50]}...")

            # Send the request to the AI and get the response.
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_message},
                    {"role": "user", "content": U_MESSAGE + chunk}
                ],
                # model=st.session_state['lm'],
                model= llm,
            )
            print(f'Model {llm} selected')
            # Extract the completion text from the response.
            completion = response.choices[0].message.content

            # Print a message to the console indicating that we received a response.
            print(f"Received response: {completion[:50]}...")

            # Try to parse the completion text as JSON and extract the listings.
            # If the parsing fails, print an error message and try to extract the listings from the completion text using a regular expression.
            try:
                parsed_chunk = json.loads(completion)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Completion content: {completion}")
                try:
                    start_index = completion.find('{"listings": [')
                    if start_index != -1:
                        listings_data = re.search(r'\{"listings": \[(.*?)\]', re.DOTALL, completion[start_index:]).group(1)
                        listings = [json.loads(listing + '}') for listing in listings_data.split('}, ')]
                        parsed_chunk = {'listings': listings}
                    else:
                        parsed_chunk = {'listings': []}
                except:
                    retry_count += 1
                    delay *= 2
                    time.sleep(delay + random.uniform(0, 1))
                    continue

            # If the parsed chunk is missing the 'listings' key, print an error message and retry.
            if 'listings' not in parsed_chunk:
                print(f"Missing 'listings' key in response: {completion}")
                retry_count += 1
                delay *= 2
                time.sleep(delay + random.uniform(0, 1))
                continue

            # Extract the prompt tokens and completion tokens from the response.
            prompt_tokens = response.usage.prompt_tokens if response.usage and response.usage.prompt_tokens else 0
            completion_tokens = response.usage.completion_tokens if response.usage and response.usage.completion_tokens else 0

            # Return the prompt tokens, completion tokens, and parsed chunk.
            return prompt_tokens, completion_tokens, parsed_chunk['listings']
        except Exception as e:
            # If an error occurs, print an error message to the console.
            print(f"Error: {e}")

            # If the error is a rate limit error, increment the rate limit retry count.
            # If we have hit the rate limit too many times, give up and return an empty list.
            if hasattr(e, 'response'):
                response = e.response
                if response is not None and response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 0))
                    print(f"Rate limit exceeded, waiting for {retry_after} seconds before retrying...")
                    time.sleep(retry_after + random.uniform(0, 1))
                    rate_limit_retries += 1
                    if rate_limit_retries >= max_rate_limit_retries:
                        break

            # Increment the retry count and wait for a short period of time before retrying.
            retry_count += 1
            delay *= 2
            time.sleep(delay + random.uniform(0, 1))

    # If we have retried too many times, return an empty list.
    return 0, 0, []

def format_data(data, DynamicListingsContainer, DynamicListingModel, selected_model):
    """
    Format data for Groq Llama3.1 70b model.

    Given a list of data, a DynamicListingsContainer, a DynamicListingModel, and a selected_model, format the data
    according to the Groq Llama3.1 70b model's requirements.

    Args:
        data (list): List of data to be formatted.
        DynamicListingsContainer: The DynamicListingsContainer class.
        DynamicListingModel: The DynamicListingModel class.
        selected_model (str): The name of the model to be used.

    Returns:
        A tuple of two items. The first item is a dictionary with a single key 'listings' whose value is a list of
        formatted data. The second item is a dictionary with two keys 'input_tokens' and 'output_tokens' whose values
        are the total number of input tokens and output tokens respectively.

    Raises:
        ValueError: If the selected_model is not 'Groq Llama3.1 70b'.
    """
    # The main entry point for formatting the data. It takes the data, a DynamicListingsContainer, a DynamicListingModel,
    # and a selected_model as input. It formats the data according to the Groq Llama3.1 70b model's requirements and
    # returns a tuple containing the formatted data and token counts.
    print("format_data function called")
    print(f"selected_model: {selected_model}")
    print(f"Data length: {len(data)}")
    print(f"Data preview: {data[:100]}")

    # Initialize an empty dictionary to store the token counts
    token_counts = {}

    # Check if the selected_model is 'Groq Llama3.1 70b'
    if selected_model == "Groq Llama3.1 70b":
        print("Correct model selected")
        # Get the Groq client object using the GROQ_KEY
        # client = groq_connection(st.session_state['groq_key'])
        client = groq_connection(st.session_state['groq_key'])
        # Generate the system message to be sent to the AI
        sys_message = generate_system_message(DynamicListingModel)
        print(f"sys_message: {sys_message}")

        # Calculate the chunk size and split the data into chunks
        chunk_size = 10000
        data_chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        print(f"data_chunks: {len(data_chunks)} chunks of size {chunk_size}")

        # Initialize an empty list to store the parsed responses
        parsed_responses = []
        # Initialize counters for the total number of input tokens and output tokens
        total_input_tokens = 0
        total_output_tokens = 0

        # Process each chunk concurrently using a ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit each chunk to be processed
            futures = [executor.submit(process_chunk, client, sys_message, chunk) for chunk in data_chunks]

            # Wait for all the chunks to be processed and collect the results
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    # Extract the input tokens, output tokens and listings from the result
                    input_tokens, output_tokens, listings = result
                    # Add the input tokens, output tokens and listings to the totals
                    total_input_tokens += input_tokens
                    total_output_tokens += output_tokens
                    parsed_responses.extend(listings)
        print("All chunks processed")

        # Combine all the parsed responses into a single dictionary
        combined_response = {'listings': parsed_responses}

        # Create a dictionary to store the token counts
        token_counts = {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens
        }

        print(f"Combined response: {combined_response}")
        print(f"Token counts: {token_counts}")

        # Return the combined response and token counts
        return combined_response, token_counts
    else:
        # If the selected_model is not 'Groq Llama3.1 70b', raise a ValueError
        print(f"Incorrect model: {selected_model}")
        raise ValueError(f"Unsupported model: {selected_model}")

def save_formatted_data(formatted_data, timestamp, output_folder='output'):
    """
    Save the formatted data to a JSON file and optionally an Excel file.

    Args:
        formatted_data (str or dict or list): The formatted data to be saved.
        timestamp (str): A timestamp to be used in the filename.
        output_folder (str, optional): The folder to save the file in. Defaults to 'output'.

    Returns:
        pandas.DataFrame: The DataFrame if it was successfully created, otherwise None.
    """
    # Create the output folder if it does not already exist
    os.makedirs(output_folder, exist_ok=True)

    # Try to parse the formatted data to a dictionary
    if isinstance(formatted_data, str):
        try:
            # Attempt to parse the string to a dictionary
            formatted_data_dict = json.loads(formatted_data)
        except json.JSONDecodeError:
            raise ValueError("The provided formatted data is a string but not valid JSON.")
    else:
        # If the formatted data is not a string, try to access its dict attribute
        # If it does not have a dict attribute, just use the formatted data as is
        formatted_data_dict = formatted_data.dict() if hasattr(formatted_data, 'dict') else formatted_data

    # Create the full path to the JSON output file
    json_output_path = os.path.join(output_folder, f'sorted_data_{timestamp}.json')

    # Write the formatted data dictionary to the JSON output file
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data_dict, f, indent=4)
    print(f"Formatted data saved to JSON at {json_output_path}")

    # Check if the formatted data dictionary is a dictionary or a list
    if isinstance(formatted_data_dict, dict):
        # If it is a dictionary, try to extract the value of the first key
        data_for_df = next(iter(formatted_data_dict.values())) if len(formatted_data_dict) == 1 else formatted_data_dict
    elif isinstance(formatted_data_dict, list):
        # If it is a list, use it as is
        data_for_df = formatted_data_dict
    else:
        # Raise a ValueError if the formatted data is neither a dictionary nor a list
        raise ValueError("Formatted data is neither a dictionary nor a list, cannot convert to DataFrame")

    try:
        # Attempt to create a DataFrame from the formatted data
        df = pd.DataFrame(data_for_df)
        print("DataFrame created successfully.")

        # Detect and convert number-like columns to numeric
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass

        # Create the full path to the Excel output file
        excel_output_path = os.path.join(output_folder, f'sorted_data_{timestamp}.xlsx')

        # Write the DataFrame to the Excel output file
        df.to_excel(excel_output_path, index=False)
        print(f"Formatted data saved to Excel at {excel_output_path}")

        # Return the DataFrame
        return df
    except Exception as e:
        # Print an error message if there was an exception
        print(f"Error creating DataFrame or saving Excel: {str(e)}")
        # Return None if there was an exception
        return None
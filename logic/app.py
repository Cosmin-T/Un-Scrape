# app.py

import streamlit as st
from logic.scraper import *
from streamlit_tags import st_tags_sidebar
import pandas as pd
import json
from datetime import datetime

def initialize_app():
    """
    Initializes the Streamlit app with a title, copyright notice, and an
    important information message about the limitations of the web scraper.

    The information message is only displayed when the user clicks the
    "Important Information" button. The message warns the user that the
    web scraper is not designed to work with websites that require user
    authentication or sessions.

    The function also sets a custom CSS style for the app, which includes a
    dark blue background, gray headings, a gradient background for the main
    content area, and custom button styles.
    """
    # The session state is used to store state between page reloads.
    # If the session state is not initialized, initialize it.
    if 'lm' not in st.session_state:
        # Set the default language model to 'llama-3.1-70b-versatile'
        st.session_state['lm'] = 'llama-3.1-70b-versatile'

    if 'wide_mode' not in st.session_state:
        # Set the default wide mode to False
        st.session_state['wide_mode'] = False

    # Set the page config only once, at the very beginning
    st.set_page_config(
        page_title="UnScraper",
        layout="wide" if st.session_state['wide_mode'] else "centered"
    )

    st.title("UnScraper")

    # Set the page title and app title.
    col1, col2 = st.columns([2,0.6])

    # Create a button to display the important information about the scraper.
    with col1:
        # Toggle to display the important information message
        info = st.toggle('Important Information')
        if info:
            # Display the important information message.
            st.warning("""
                **Please Note:**

                UnScrape is not designed to scrape websites that require user authentication or sessions.

                **Examples of incompatible websites:**
                * E-commerce sites like Amazon
                * Social media platforms
                * Online banking and payment gateways
                * Websites with restricted content access

                **Alternatives for scraping these websites:**
                * Use automation frameworks like Selenium or Puppeteer
                * Utilize browser emulators
                * Leverage APIs or developer tools
            """)

    with col2:
        # Toggle to switch to wide mode
        wide = st.toggle('Wide Mode')
        if wide and not st.session_state['wide_mode']:
            # Set the wide mode to True and rerun the app
            st.session_state['wide_mode'] = True
            st.rerun()
        elif not wide and st.session_state['wide_mode']:
            # Set the wide mode to False and rerun the app
            st.session_state['wide_mode'] = False
            st.rerun()

    # Display the copyright information.
    st.markdown('<p style="font-size: small;">© 2024 Developed by CosminT. All rights reserved.</p>', unsafe_allow_html=True)

    # Add a horizontal line to separate the title from the rest of the app.
    st.markdown("---")

    # Set the custom CSS style for the app.
    custom_css = """
    <style>
        /* Set the background color to a dark blue. */
        body {
            background-color: #002b36;
        }

        /* Set the color and font weight of the headings. */
        h1, h2, h3, h4, h5, h6 {
            color: #999;
            font-weight: 500;
            transition: color 0.3s;
        }

        /* Set the background color of the main content area to a gradient. */
        .stApp {
            background: linear-gradient(200deg, #002b36 -50%, #1e1e1e 75%);
            border-radius: 10px;
            box-shadow: 3px 3px 20px rgba(0, 0, 0, 0.3);
            padding: 50px;
        }

        /* Set the style of the buttons. */
        .stButton>button {
            background-color: #6a2336;
            color: #FFF;
            border: none;
            border-radius: 12px;
        }

        /* Change the button color on hover. */
        .stButton>button:hover {
            background-color: #5E0000;
        }

        /* Customize the toggle button */
        .st-ck {
            background-color: #6a2336;
            border-radius: 12px;
            padding: 8px 10px;
        }
    </style>
    """
    # Apply the custom CSS style to the app.
    st.markdown(custom_css, unsafe_allow_html=True)

def create_sidebar():
    """
    Creates a custom sidebar with a title, a button to buy the author a coffee,
    a divider, and a text input for the user to enter a URL.

    The sidebar also includes a tag input field for the user to enter fields/tags
    to scrape from the given URL.

    Returns:
        tuple: A tuple containing the URL input and the tags input.
    """

    # The URL to open when the user clicks on the button.
    url = "https://revolut.me/cosminhbs7"

    # Custom styles for the sidebar.
    button_style = """
    <style>
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #262730;
    }
    .custom-button {
        display: inline-block;
        padding: 12px 25px;
        font-size: 18px;
        font-weight: bold;
        color: white;
        background-color: #262730;
        border: none;
        border-radius: 8px;
        text-align: center;
        text-decoration: none;
        width: 100%;
        transition: background-color 0.3s ease;
    }
    .custom-button:hover {
        background-color: #6a2336;
        color: white;
    }
    .divider {
        border: none;
        border-top: 2px solid #e0e0e0;
        margin: 20px 0;
    }
    </style>
    """
    # Apply the custom CSS style to the sidebar.
    st.sidebar.markdown(button_style, unsafe_allow_html=True)
    # Display the title of the app.
    st.sidebar.markdown('<div class="sidebar-title">UnScraper</div>', unsafe_allow_html=True)
    # Display the button to buy the author a coffee.
    st.sidebar.markdown(f'<a href="{url}" target="_blank" class="custom-button">Buy Me A Coffee</a>', unsafe_allow_html=True)
    # Display a dividing line.
    st.sidebar.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Add some empty space to the sidebar.
    for i in range(4):
        st.sidebar.markdown('#')

    # Add Groq API
    groq_api = st.sidebar.text_input("Enter Groq API Key", placeholder="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")
    try:
        if re.match(r'^[A-Za-z0-9_]{43,}$', groq_api):
            # Load api into session state
            st.session_state['groq_key'] = groq_api
        else:
            st.sidebar.error("Invalid API Key format. Please enter a valid key.")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

    # Get the URL from the user.
    url_input = st.sidebar.text_input("Enter URL", placeholder="https://example.com")

    # Get the fields/tags from the user.
    tags = st_tags_sidebar(
        label='Enter Fields/Tags',
        text='Press enter to add tag',
        maxtags=-1,  # -1 for unlimited
        key='tags_input'
    )

    # Return the URL and the fields/tags.
    return url_input, tags

def loading():
    st.markdown(
        """
        <style>
        .stProgress > div > div > div > div {
            background-color: #6a2336;
            height: 15px;
            border-radius: 2px;
        }
        .stProgress > div {
            width: 10%;
            margin: auto;
            margin-left: 0;
        }
        .loading-dots {
            font-size: 1rem;
        }
        .loading-dots::after {
            content: '';
            animation: loading-dots 2s infinite;
        }
        @keyframes loading-dots {
            0% { content: ''; }
            25% { content: '.'; }
            50% { content: '..'; }
            75% { content: '...'; }
            100% { content: ''; }
        }
         .stProgress-loaded > div > div > div > div {
            background-color: #6a2336; /* Here's the color change */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    text = st.markdown('###### <span class="loading-dots">Scraping Data</span>', unsafe_allow_html=True)
    progress_bar = st.progress(0)

    return text, progress_bar

def perform_scrape(url_input, fields):
    """
    Fetches the HTML of the given URL, converts it to Markdown and then parses it
    into a structured data format based on the given fields/tags.

    Args:
        url_input (str): The URL to fetch.
        fields (list[str]): The fields/tags to extract from the page.

    Returns:
        tuple: A tuple containing the following items:
            - df (pd.DataFrame): The structured data in a pandas DataFrame.
            - formatted_data (list[dict]): The structured data in a list of dictionaries.
            - markdown (str): The Markdown representation of the page.
            - timestamp (str): The timestamp of when the data was scraped.
    """

    # Initiate progress bar
    text, progress_bar = loading()

    # Get the current timestamp.
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    progress_bar.progress(0.125)
    # Fetch the HTML of the given URL.
    raw_html = fetch_html_playwright(url_input)
    progress_bar.progress(0.25)
    # Convert the HTML to Markdown.
    markdown = html_to_markdown_with_readability(raw_html)
    progress_bar.progress(0.375)
    # Save the raw Markdown data to a file.
    save_raw_data(markdown, timestamp)
    progress_bar.progress(0.50)

    # Create a dynamic Pydantic model based on the given fields/tags.
    DynamicListingModel = create_dynamic_listing_model(fields)
    progress_bar.progress(0.625)
    # Create a dynamic Pydantic model that contains a field called "listings" which is a list of the given listing model.
    DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
    progress_bar.progress(0.75)
    # Parse the Markdown to a structured data format and get the number of input and output tokens used by the AI.
    formatted_data, token_counts = format_data(markdown, DynamicListingsContainer, DynamicListingModel, "Groq Llama3.1 70b")
    st.info(f"Input tokens: {token_counts['input_tokens']}")
    st.info(f"Output tokens: {token_counts['output_tokens']}")
    progress_bar.progress(0.875)

    # Save the structured data to a file.
    df = save_formatted_data(formatted_data, timestamp)
    progress_bar.progress(1)


    progress_bar.progress(1.0)
    text.empty()
    progress_bar.empty()

    # Return the structured data, the Markdown representation of the page and the timestamp.
    return df, formatted_data, markdown, timestamp

def create_main_page(df, formatted_data, markdown, timestamp):
    """
    Creates the main page of the Streamlit app with download buttons for the data in different formats.

    Parameters
    ----------
    df : pd.DataFrame
        The structured data in a pandas DataFrame.
    formatted_data : list[dict] or str
        The structured data in a list of dictionaries or as a string.
    markdown : str
        The Markdown representation of the page.
    timestamp : str
        The timestamp of when the data was scraped.
    """
    # Apply CSS to match the download buttons with normal Streamlit buttons
    st.markdown("""
        <style>
        .stDownloadButton > button {
            font-family: "Source Sans Pro", sans-serif;
            font-weight: 400;
            font-size: inherit;
            line-height: 1.6;
            color: #ffffff;
            background-color: transparent;
            border: none;
            padding: 0.25rem 0.75rem;
            cursor: pointer;
            width: auto;
            user-select: none;
            background-color: rgb(106, 35, 54); /* Changed color to a blue shade */
            border: 1px solid rgba(0, 0, 0, 0.2);
            border-radius: 0.7rem; /* Increased border radius for more rounded corners */
            transition: border 200ms ease 0s, background-color 200ms ease 0s;
        }

        .stDownloadButton > button:hover {
            background-color: #5E0000;
        }

        /* Ensure dark mode compatibility */
        @media (prefers-color-scheme: dark) {
            .stDownloadButton > button {
                color-scheme: dark;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    # Create the main page of the Streamlit app with three columns.
    # The first column contains a download button for the JSON data.
    # The second column contains a download button for the CSV data.
    # The third column contains a download button for the Markdown text.
    col1, col2, col3 = st.columns([1, 1, 0.62])

    # The first column contains a download button for the JSON data.
    with col1:
        # Use the json.dumps() function to convert the formatted_data to a string.
        # The indent parameter is set to 4 to pretty-print the JSON data.
        st.download_button(
            "Download RAW JSON",
            data=json.dumps(formatted_data.dict() if hasattr(formatted_data, 'dict') else formatted_data, indent=4),
            file_name=f"{timestamp}_data.json",
        )

    # The second column contains a download button for the CSV data.
    with col2:
        # If the formatted_data is a string, convert it to a dictionary.
        if isinstance(formatted_data, str):
            data_dict = json.loads(formatted_data)
        else:
            # If the formatted_data is a list of dictionaries, use the first key.
            data_dict = formatted_data.dict() if hasattr(formatted_data, 'dict') else formatted_data

        # Get the first key of the dictionary.
        first_key = next(iter(data_dict))

        # Get the value of the first key.
        main_data = data_dict[first_key]

        # Convert the main_data to a pandas DataFrame.
        df_csv = pd.DataFrame(main_data)

        # Use the to_csv() method to convert the DataFrame to a CSV string.
        # The index parameter is set to False to exclude the index column.
        st.download_button(
            "Download RAW CSV",
            data=df_csv.to_csv(index=False),
            file_name=f"{timestamp}_data.csv",
        )

    # The third column contains a download button for the Markdown text.
    with col3:
        # Use the st.download_button() function to create a download button for the Markdown text.

        st.download_button(
            "Download RAW MD",
            data=markdown,
            file_name=f"{timestamp}_data.md",
        )

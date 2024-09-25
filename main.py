# main.py

from logic.app import *
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit_extras.dataframe_explorer import dataframe_explorer

def display_df(df):
    """
    Displays a DataFrame in a Streamlit app, allowing the user to select which columns to display
    and control the height of the DataFrame itself.
    If df is None, do nothing.
    Otherwise, create a copy of the DataFrame to avoid modifying the original,
    create a column selection section, reset the index before displaying,
    and finally display the DataFrame with user-specified height.
    Args:
        df (pandas.DataFrame): DataFrame to display.
    Returns:
        None
    """
    if df is None:
        return
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Ensure all columns are of a compatible data type
    for col in df_copy.columns:
        if df_copy[col].dtype == object:
            try:
                df_copy[col] = pd.to_numeric(df_copy[col])
            except ValueError:
                df_copy[col] = df_copy[col].astype(str)

    # Create a column selection section
    with st.expander("Column Selection", expanded=True):
        # Select which columns to display
        selected_columns = st.multiselect("Select columns to display", df_copy.columns, default=df_copy.columns.tolist())
        # Apply the selection
        df_copy = df_copy[selected_columns]

    # Reset the index before displaying
    df_copy = df_copy.reset_index(drop=True)
    # Get the height from the user
    height = st.slider("Set DataFrame height", min_value=100, max_value=5000, value=500, step=10)
    # Display the DataFrame with the specified height
    st.dataframe(df_copy, use_container_width=True, height=height)

def main():
    """
    Entry point for the Streamlit application.

    This function is the main entry point for the Streamlit application. It
    initializes the application, creates a sidebar, and then displays the
    scraped data in the main page.

    This function does not take any arguments and does not return any values.
    """
    try:
        # Initialize the Streamlit application
        initialize_app()

        # Create a sidebar
        url_input, fields = create_sidebar()

        # Input validation
        if not url_input:
            st.sidebar.error("Please add a URL.")
            return

        # Initialize a session state variable
        if 'perform_scrape' not in st.session_state:
            st.session_state['perform_scrape'] = False

        # Run the scraper if the user clicks the "Scrape" button
        if st.sidebar.button("Scrape"):
            # with st.spinner('Processing Scraped Data...'):
            # Check if URL is valid
            try:
                if not url_input.startswith("http"):
                    raise ValueError("Invalid URL. URL must start with http:// or https://")
                result = url_input.split("://")
                if len(result) != 2:
                    raise ValueError("Invalid URL. URL format is incorrect.")
            except ValueError as e:
                st.sidebar.error(f"Error with URL: {e}")
                return
            except Exception as e:
                st.sidebar.error(f"Error processing URL: {e}")
                return

            # Run the scraper
            try:
                st.session_state['results'] = perform_scrape(url_input, fields)
            except Exception as e:
                st.sidebar.error(f"Error scraping the website: {e}")
                return
            # Set the session state to True to indicate that the scraper has run
            st.session_state['perform_scrape'] = True

        # If the scraper has run, display the scraped data
        if st.session_state.get('perform_scrape'):
            try:
                # Display the scraped data in the main page
                df, formatted_data, markdown, timestamp = st.session_state['results']

                # Create an expanded container
                with st.container():
                    display_df(df)

                # Display other elements outside the scrollable container
                create_main_page(df, formatted_data, markdown, timestamp)

            except Exception as e:
                st.sidebar.error(f"Error processing scraped data: {e}")

    except Exception as e:
        st.sidebar.error(f"Unknown Error: {e}")


# Check if this script is being run directly (not being imported as a module)
if __name__ == "__main__":
    # Call the main function to start the execution of the script
    main()

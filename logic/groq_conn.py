# groq.py

from logic.util import *
from groq import Groq

def groq_connection(api_key):
    """
    Establishes a connection with the Groq API using the given API key.

    Returns the Groq client object if the connection is successful, otherwise None.
    """
    try:
        client = Groq(
            api_key=api_key,
        )
        print('Groq Connection Established Successfully')
        # models = client.models.list()
        # print(f"Available Models: {models}")

        return client
    except Exception as e:
        print(f"Error creating Groq client: {e}")
        return None
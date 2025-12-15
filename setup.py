import os
import getpass

# Set the Groq API Key
# IMPORTANT: This will prompt you to enter the key securely.
# Press Enter after pasting your key.
if 'GROQ_API_KEY' not in os.environ:
    os.environ['GROQ_API_KEY'] = getpass.getpass("Enter your Groq API Key: ")

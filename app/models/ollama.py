import requests
import json

def summarize_email_llama(email_content):
    """
    Summarizes the given email content using Ollama.
    Args:
    - email_content (str): The content of the email to summarize.
    Returns:
    - str: The summarized text of the email.
    """
    # Ollama API endpoint (default local installation)
    url = "http://localhost:11434/api/generate"
    
    # Create a prompt that guides the model for business email summarization
    prompt = f"""
    Please summarize this business email concisely, focusing on:
    - Main topic/request
    - Key details and features
    - Action items and deadlines
    - Any special offers
    
    Email content:
    {email_content}
    
    Provide a clear, concise summary that captures all important points.
    """
    
    # Prepare the request data
    data = {
        "model": "mistral",  # You can change this to any model you have pulled in Ollama
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,  # Lower temperature for more focused output
            "top_p": 0.9,
            "top_k": 40
        }
    }
    
    try:
        # Make the API call
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the response
        result = response.json()
        summary = result['response']
        
        # Clean up the summary
        cleaned_summary = summary.strip()
        
        return cleaned_summary
        
    except requests.exceptions.RequestException as e:
        return f"Error generating summary: {str(e)}"
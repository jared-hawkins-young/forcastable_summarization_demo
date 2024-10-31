import psutil
import time
import torch
from transformers import pipeline

def summarize_email(email_content):
    """
    Summarizes the email content and tracks CPU, memory, and GPU usage.

    Args:
        email_content (str): The content of the email to summarize.

    Returns:
        str: The summarized text of the email.
    """
    # Track start time
    start_time = time.time()

    # Log initial CPU and memory usage
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = psutil.virtual_memory().used / (1024 ** 3)  # Convert to GB

    # Initialize summarizer
    summarizer = pipeline('summarization', model="google/pegasus-cnn_dailymail")

    # Define summary constraints
    max_length = 400
    min_length = 100

    # Generate summary
    summary = summarizer(
        email_content,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )[0]['summary_text']

    # Track end time
    end_time = time.time()

    # Log final CPU, memory, and GPU usage
    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = psutil.virtual_memory().used / (1024 ** 3)  # Convert to GB
    gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else "N/A"

    # Display usage stats
    print(f"CPU Usage: {cpu_after - cpu_before}%")
    print(f"Memory Before: {mem_before:.2f} GB, After: {mem_after:.2f} GB")
    if gpu_memory != "N/A":
        print(f"GPU Memory Used: {gpu_memory:.2f} GB")
    print(f"Processing Time: {end_time - start_time:.2f} seconds")

    return summary

def summarize_email2(email_content):
    """
    Summarizes the given email content with enhanced focus on business context and details.

    Args:
    - email_content (str): The content of the email to summarize.

    Returns:
    - str: The summarized text of the email.
    """
    # Initialize summarizer - using BART model which is better at maintaining details
    summarizer = pipeline('summarization', model="google/pegasus-cnn_dailymail")

    # Create a more specific prompt that forces detail retention
    prompted_input = f"""
        Business Email Summary:

        {email_content}

        Key Points to Include: Purpose, Products, Features, Offers, Next Steps
    """

    # Adjust length constraints to allow for more detail
    content_length = len(email_content.split())
    # Increasing the ratio to allow for more detailed summaries
    max_length = min(int(content_length * 0.7), 400)  # 70% of original, max 400 words
    min_length = max(int(content_length * 0.4), 100)  # 40% of original, min 100 words

    # Generate summary with parameters optimized for detail retention
    summary = summarizer(
        prompted_input,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        num_beams=4,
        repetition_penalty=1.2,  # Helps avoid repetitive text
        length_penalty=2.0,      # Encourages slightly longer summaries
        early_stopping=True
    )

    return summary[0]['summary_text']

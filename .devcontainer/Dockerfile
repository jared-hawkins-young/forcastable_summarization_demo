# Start from a base Ubuntu image
FROM ubuntu:20.04

# Fix GPG keys and sources to resolve invalid signature issues
RUN apt-get update && apt-get install -y gnupg && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32 && \
    sed -i 's|http://archive.ubuntu.com/ubuntu/|http://mirror.math.princeton.edu/pub/ubuntu/|g' /etc/apt/sources.list && \
    apt-get update

# Install system dependencies
RUN apt-get install -y \
    curl \
    sudo \
    python3 \
    python3-pip

# Install Ollama
RUN curl -sSfL https://ollama.ai/install.sh | sh

# Expose the default port Ollama uses
EXPOSE 11434

# Start Ollama in server mode
CMD ["ollama", "serve"]
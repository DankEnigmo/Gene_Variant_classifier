FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    perl \
    python3 \
    python3-pip \
    build-essential \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    git \
    gzip \
    && rm -rf /var/lib/apt/lists/*

# Create Python symlink
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Install Python packages
RUN pip3 install pandas numpy vcfpy scikit-learn xgboost==2.0.3 scipy optuna matplotlib imbalanced-learn

# Create data directory
RUN mkdir -p /app/data

# Copy application files
COPY src/ /app/src/
COPY data/ /app/data/

# Create and set up the run script
RUN echo '#!/bin/bash\n\
python3 src/main.py' > /app/run.sh \
    && chmod +x /app/run.sh

# Default command runs your Python script
CMD ["/app/run.sh"]

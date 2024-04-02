# Use an official Python runtime as a parent image
FROM python:3.11

# Install R, which is necessary for running R scripts
# Install system libraries required for R and Python
RUN apt-get update && apt-get install -y \
    libfontconfig1-dev \
    libfreetype6-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    build-essential \
    r-base \
    pandoc \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container to /footy-tipper
WORKDIR /footy-tipper

# Copy the current directory contents into the container at /footy-tipper
COPY . /footy-tipper

# Create a Python virtual environment and install Python packages
# Note: requirements.txt is assumed to be at the root of the project
RUN python3 -m venv footyenv
ENV PATH="/footy-tipper/footyenv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Install any needed packages specified in install.R
# Note: install.R is assumed to be at the root of the project
RUN Rscript install.R

# Make port 80 available to the world outside this container
EXPOSE 80

# Set an environment variable to indicate that the application is running in Docker
ENV DOCKER=true

# Adjust the CMD command to run scripts from the pipeline directory
CMD Rscript pipeline/data-prep.R && python pipeline/train.py && python pipeline/inference.py && python pipeline/send.py

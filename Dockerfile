# Use an official Python runtime as a parent image
FROM python:3.11

# Install system libraries required for R and Python
RUN apt-get update && apt-get install -y \
    libfontconfig1-dev \
    libfreetype6-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    build-essential \
    r-base

# Install pandoc
RUN apt-get update \
 && apt-get install -y --no-install-recommends pandoc \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container to /footy-tipper
WORKDIR /footy-tipper

# Copy the current directory contents into the container at /footy-tipper
COPY . /footy-tipper

# Create a Python virtual environment and install Python packages
RUN python3 -m venv footyenv
ENV PATH="/footy-tipper/footyenv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Install any needed packages specified in install.R
RUN Rscript install.R

# Make port 80 available to the world outside this container
EXPOSE 80

# When running in container set DOCKER variable to be true
ENV DOCKER true

# Add a new argument for which process to run
ARG PROCESS

# Use the PROCESS argument to set an environment variable
ENV PROCESS=${PROCESS}

# Run appropriate Python script based on PROCESS when the container launches
CMD ["python", "${PROCESS}.py"]

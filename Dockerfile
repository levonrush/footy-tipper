# Use an official Python runtime as a parent image
FROM python:3.11

# Install R (Debian's r-base; trixie ships R >= 4.5) and system libraries
# required for R and Python, including cmake
RUN apt-get update && apt-get install -y \
    libfontconfig1-dev \
    libfreetype6-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    build-essential \
    r-base \
    pandoc \
    cmake \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container to /footy-tipper
WORKDIR /footy-tipper

# Copy the project files except the ones defined in .dockerignore
COPY . /footy-tipper

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# R packages live at a fixed path so runtime HOME changes (e.g. GitHub Actions
# containers set HOME=/github/home) cannot hide them from .libPaths().
ENV R_LIBS_USER=/opt/r-library

# Install any needed packages specified in install.R
RUN Rscript install.R

# Make port 80 available to the world outside this container
EXPOSE 80

# Set an environment variable to indicate that the application is running in Docker
ENV DOCKER=true

# Default command for the container
CMD ["python", "-m", "pipeline.cli", "--help"]
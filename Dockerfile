# Use an official Python runtime as a parent image
# FROM python:3.11
FROM python:3.9.6

# # Add CRAN repository for newer R versions and add keys
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     software-properties-common dirmngr gnupg \
#     && wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc \
#     && add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"


# Install R and system libraries required for R and Python
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

# Install R separately without specifying a CRAN repository
# Note: This installs the version of R available in Debian Bullseye's repositories
RUN apt-get update && apt-get install -y r-base

# Set the working directory in the container to /footy-tipper
WORKDIR /footy-tipper

# Copy the project files except the ones defined in .dockerignore
COPY . /footy-tipper

# Create a Python virtual environment and install Python packages
RUN python3 -m venv footyenv
ENV PATH="/footy-tipper/footyenv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Install any needed packages specified in install.R
RUN Rscript install.R

# Make port 80 available to the world outside this container
EXPOSE 80

# Set an environment variable to indicate that the application is running in Docker
ENV DOCKER=true

# Run the footy-tipper.py script
CMD ["python", "footy-tipper.py"]

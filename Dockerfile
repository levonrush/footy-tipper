# Use an official R runtime as a parent image
FROM r-base:4.2.3

# Install system libraries
RUN apt-get update && apt-get install -y \
    libfontconfig1-dev \
    libfreetype6-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential

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

# when running in container set DOCKER variable to be trie
ENV DOCKER true

# Run footy-tipper.R when the container launches
CMD ["Rscript", "footy-tipper.R"]

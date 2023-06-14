# Use an official R runtime as a parent image
FROM r-base:latest

# Install pandoc
RUN apt-get update \
 && apt-get install -y --no-install-recommends pandoc \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install system libraries
RUN apt-get update && apt-get install -y \
    libfontconfig1-dev \
    libfreetype6-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev

# Set the working directory in the container to /footy-tipper
WORKDIR /footy-tipper

# Copy the current directory contents into the container at /footy-tipper
COPY . /footy-tipper

COPY footy-tipper-c5bcb9639ee2.json /footy-tipper/service-account-token.json

# Install any needed packages specified in install.R
RUN Rscript install.R

# Make port 80 available to the world outside this container
EXPOSE 80

# set the default type of run to test (ie. not prod)
ENV PROD_RUN F

# Render footy-tipper.Rmd when the container launches
CMD ["R", "-e", "rmarkdown::render('footy-tipper.Rmd')"]

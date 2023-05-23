# Use an official R runtime as a parent image
FROM r-base:latest

# Install pandoc
RUN apt-get update \
 && apt-get install -y --no-install-recommends pandoc \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container to /footy-tipper
WORKDIR /footy-tipper

# Copy the current directory contents into the container at /footy-tipper
COPY . /footy-tipper

COPY /Users/levonrush/Downloads/footy-tipper-c5bcb9639ee2.json ;2D/footy-tipper/service-account-token.json

# Install any needed packages specified in install.R
RUN Rscript install.R

# Make port 80 available to the world outside this container
EXPOSE 80

# Render footy-tipper.Rmd when the container launches
CMD ["R", "-e", "rmarkdown::render('footy-tipper.Rmd')"]

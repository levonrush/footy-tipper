# Use an official R runtime as a parent image
FROM r-base:latest

# Set the working directory in the container to /footy-tipper
WORKDIR /footy-tipper

# Copy the current directory contents into the container at /footy-tipper
COPY . /footy-tipper

# Install any needed packages specified in install.R
RUN Rscript install.R

# Make port 80 available to the world outside this container
EXPOSE 80

# Run footy-tipper.Rmd when the container launches
CMD ["R", "-e", "rmarkdown::render('footy-tipper.Rmd')"]

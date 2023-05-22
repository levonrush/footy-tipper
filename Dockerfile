# Use an official R runtime as a parent image
FROM r-base:latest

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in install.R
RUN Rscript install.R

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.R when the container launches
CMD ["Rscript", "footyTipper.Rmd"]

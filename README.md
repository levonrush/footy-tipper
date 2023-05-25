# Footy-Tipper: A machine learning approach to winning the pub tipping comp

Footy-Tipper is an open-source Rugby League prediction engine designed to predict the outcomes of National Rugby League (NRL) matches. Initially created as a data science project for a pub tipping competition, this tool is now used by NRL fans who want to gain an edge in their own tipping competitions or just want to understand the game better.

## Background and Motivation

Footy-Tipper was created by Levon Rush, an NRL enthusiast, who combines his love for the sport and data science in this project. Levon began the project as a way to gain an advantage in his local pub's footy-tipping competition at the Seven Seas Hotel. The idea was simple: use machine learning to predict match outcomes more accurately than the average punter.

The project began with a simple question, "Can machine learning improve my footy tips?" As it turns out, the answer was a resounding "yes". You can read more about the background and motivation of the project in Levon's [blog post on Medium](https://medium.com/@levonrush/the-footy-tipper-a-machine-learning-approach-to-winning-the-pub-tipping-comp-dc07a7325292).

## How it works

Footy-Tipper leverages freely available NRL data, performs data cleaning and feature engineering, and applies machine learning algorithms to predict the outcomes of NRL matches. The pipeline of the project is implemented in R and wrapped in a Docker container for portability and ease of deployment.

## Prerequisites

- Docker
- R and Visual Studio Code (for development and debugging)
- Google service account for Google Drive authentication (service-account-token.json)
- Google Cloud Platform (for computation)

## Usage

To run Footy-Tipper in a production environment:

1. Clone this repository.
```
git clone https://github.com/<your_username>/footy-tipper.git
```

2. Go to the project's directory.
```
cd footy-tipper
```

3. Build the Docker image.
```
docker build -t footy-tipper .
```

4. Run the Docker container with the environment variables defined in a `secrets.env` file and map the internal Docker port 80 to your host's port 4000.
```
docker run --env-file secrets.env -e PROD_RUN="F" -p 4000:80 footy-tipper
```

To run Footy-Tipper in VS Code for development and debugging:

1. Open the project in VS Code.
2. Set the environment variables in a `.env` file or manually in your VS Code session.
3. Run the pipeline.

## Contributing

Footy-Tipper welcomes contributions from the community. Please check the issues section of the repository to see how you can contribute.

## Contact

To obtain the project's secrets or for any questions or comments related to this project, you can reach out to Levon at levon_rush@hotmail.com

For more of the blog itself you can find it on [Levon's Medium profile](https://medium.com/@levonrush).

## Acknowledgements
Special thanks to Seven Seas Hotel for motivating this project, Kate for telling me to have a portfolio piece, Victoria for the emotional support, and ChatGPT for writing this readme.

# The Footy-Tipper: A Machine Learning Approach to Winning the Pub Tipping Comp

The Footy-Tipper is an open-source Rugby League prediction engine designed to predict the outcomes of National Rugby League (NRL) matches. This project artfully blends the capabilities of R, Python, and SQL to develop a wholistic data science product. R is used for constructing data pipelines and researching Rugby League behaviors, Python for machine learning modeling, and SQL for efficient data management across platforms.

A development blog, titled "The Footy Tipper," provides detailed insights into the progress and findings of this project. You can read the first edition of the blog [here](https://medium.com/@levonrush/the-footy-tipper-a-machine-learning-approach-to-winning-the-pub-tipping-comp-dc07a7325292).

## How it Works

Footy-Tipper leverages freely available NRL data, performs data cleaning, and feature engineering in R to investigate and understand Rugby League behaviors. Python, being versatile and having a rich set of libraries for machine learning, data manipulation, and interacting with APIs, is employed for constructing predictive models, generating and uploading predictions to Google Drive, generating game synopsis using OpenAI, and sending out automated emails with game predictions. SQL plays an integral role in managing and transferring data between different platforms and environments. The synergy of these technologies results in a powerful prediction engine. The pipeline of the project is wrapped in a Docker container for portability and ease of deployment.

## Prerequisites

- Docker installed and running on your machine.
- Google service account for Google Drive authentication (`service-account-token.json`).
- (Optional) R, Python and Visual Studio Code, if you want to develop or debug locally.
- (Optional) Google Cloud Platform account for computation.

## Usage

### Using Docker

1. Clone this repository.
    ```
    git clone https://github.com/levonrush/footy-tipper.git
    ```

2. Navigate to the project's directory.
    ```
    cd footy-tipper
    ```

3. Build the Docker image.
    ```
    docker build -t footy-tipper .
    ```

4. Run the Docker container, replacing `<your_host_port>` with the port number you want to use on your host machine (e.g., 4000).
    ```
    docker run -p <your_host_port>:80 footy-tipper
    ```

### For Development and Debugging

1. Open the project in R Studio or Visual Studio Code.
2. Set the environment variables in a `.env` file or manually in your R session.
3. Run the pipeline by executing `footy-tipper.R`.

## Contributing

Footy-Tipper welcomes contributions from the community. Please check the issues section of the repository to see how you can contribute.

## Contact

To obtain the project's secrets or for any questions or comments related to this project, please reach out via the repository's issues section.

## Acknowledgements
Special thanks to Seven Seas Hotel for motivating this project, Kate for telling me to make myself a portfolio piece, Victoria and Ernie for the emotional support, and ChatGPT for writing this readme.

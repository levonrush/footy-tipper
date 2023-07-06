# The Footy-Tipper: A Machine Learning Approach to Winning the Pub Tipping Comp

The Footy-Tipper is an open-source Rugby League prediction engine designed to predict the outcomes of National Rugby League (NRL) matches. This project artfully blends the capabilities of R, Python, and SQL to develop a wholistic data science product. R is used for constructing data pipelines and researching Rugby League behaviors, Python for machine learning modeling, and SQL for efficient data management across platforms.

A development blog, titled "The Footy Tipper," provides detailed insights into the progress and findings of this project. You can read the first edition of the blog [here](https://medium.com/@levonrush/the-footy-tipper-a-machine-learning-approach-to-winning-the-pub-tipping-comp-dc07a7325292).

## How it Works

Footy-Tipper operates by harnessing freely available NRL data and implementing a series of scripts and notebooks in a systematic pipeline. The journey commences with the `data-prep.R` script, situated in the 'data-prep' folder. This script is charged with data cleaning and feature engineering, furthering our grasp of Rugby League behaviors.

Subsequent to the data preparation, the pipeline moves forward to the model development phase. This is executed in the `model-training.ipynb` notebook situated in the 'model-training' folder. Python, renowned for its flexibility and extensive suite of machine learning libraries, is utilized for constructing robust predictive models from the preprocessed data.

Upon successful model training, the pipeline proceeds to the `send_predictions.ipynb` notebook in the 'use-predictions' folder. Here, the model's predictions are generated and uploaded to Google Drive. In a unique twist, an email synopsis, crafted in the persona of Reg Regan, is generated using OpenAI's language model. These predictions, embellished with Reg Regan's characteristic flair, are dispatched via automated emails, ensuring all recipients receive the latest forecasts with an enjoyable twist.

In addition to the operational pipeline, Footy-Tipper also houses an extensive body of research that was integral to its development. This research, stored in the 'research' folder, encompasses various preliminary analyses, data explorations, and experimental model iterations. These research artifacts serve a dual purpose - they are a testament to the methodical process of building Footy-Tipper and also act as a resource for future enhancements. By maintaining transparency and retaining a rich record of the project's development journey, we ensure that Footy-Tipper remains open to continuous evolution and improvement, driven by the latest advances in predictive analytics and machine learning research.

Throughout this pipeline, SQL plays a vital role in data management and transition across various platforms and environments. Furthermore, Docker encapsulates the entire pipeline, ensuring portability and facilitating easy deployment. In essence, the synergy of R, Python, SQL, and Docker, coupled with the entertaining narrative of Reg Regan, coalesce to create the compelling prediction engine that is Footy-Tipper.

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

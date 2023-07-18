# The Footy-Tipper: A Machine Learning Approach to Winning the Pub Tipping Comp

The Footy-Tipper is an open-source Rugby League prediction engine designed to predict the outcomes of National Rugby League (NRL) matches. This project artfully blends the capabilities of R, Python, and SQL to develop a wholistic data science product. R is used for constructing data pipelines and researching Rugby League behaviors, Python for machine learning modeling, and SQL for efficient data management across platforms.

A development blog, titled "The Footy Tipper," provides detailed insights into the progress and findings of this project. You can read the first edition of the blog [here](https://medium.com/@levonrush/the-footy-tipper-a-machine-learning-approach-to-winning-the-pub-tipping-comp-dc07a7325292).

## How it Works

**Model Building:** This process starts with the `data-prep.R` script, situated in the 'data-prep' folder, which is responsible for data cleaning and feature engineering. Following the data preparation, the model development phase is executed in the `model-training.ipynb` notebook situated in the 'model-training' folder. Python, renowned for its flexibility and extensive suite of machine learning libraries, is used for constructing robust predictive models from the preprocessed data.

**Model Prediction:** This process begins again with the `data-prep.R` script to ensure that the most recent data is used. It then proceeds to the `model-prediction.ipynb` and `send_predictions.ipynb` notebooks in the 'model-prediction' and 'use-predictions' folders respectively. Here, the model's predictions are generated, uploaded to Google Drive, and dispatched via automated emails.

Throughout these processes, SQL plays a vital role in data management and transition across various platforms and environments. Docker encapsulates both processes, ensuring portability and facilitating easy deployment.

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

3. To build the Docker image for the model building process:
    ```
    docker build --build-arg PROCESS=model_building -t my-footy-tipper-building .
    ```

4. To build the Docker image for the model prediction process:
    ```
    docker build --build-arg PROCESS=model_prediction -t my-footy-tipper-prediction .
    ```

5. Run the Docker container, replacing `<your_host_port>` with the port number you want to use on your host machine (e.g., 4000), and `<image>` with the Docker image you want to run (`my-footy-tipper-building` or `my-footy-tipper-prediction`).
    ```
    docker run -p <your_host_port>:80 <image>
    ```

### For Development and Debugging

1. Open the project in your preferred code editor.
2. If needed, set environment variables in a `.env` file or manually in your Python or R session.
3. Run the `data-prep.R` script located in the 'data-prep' folder for data cleaning and feature engineering.
4. For model building, open and execute the `model-training.ipynb` notebook situated in the 'model-training' folder.
5. For model prediction, execute the `model-prediction.ipynb` notebook in the 'model-prediction' folder, followed by the `send_predictions.ipynb` notebook in the 'use-predictions' folder. The latter notebook sends out the model's predictions.
6. If Docker is used, ensure to build and run the Docker image as necessary.

Note: Ensure your Python and R environments have all necessary packages installed to run the scripts and notebooks.

## Contributing

Footy-Tipper welcomes contributions from the community. Please check the issues section of the repository to see how you can contribute.

## Contact

To obtain the project's secrets or for any questions or comments related to this project, please reach out via the repository's issues section.

## Acknowledgements
Special thanks to Seven Seas Hotel for motivating this project, Kate for telling me to make myself a portfolio piece, Victoria and Ernie for the emotional support, and ChatGPT for writing this readme.

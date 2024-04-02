# The Footy-Tipper: A Machine Learning Approach to Winning the Pub Tipping Comp

The Footy-Tipper is an open-source Rugby League prediction engine designed to predict the outcomes of National Rugby League (NRL) matches. This project artfully blends the capabilities of R, Python, and SQL to develop a wholistic data science product. R is used for constructing data pipelines and researching Rugby League behaviors, Python for machine learning modeling, and SQL for efficient data management across platforms.

![Footy Tipper Logo](/images/footy-tipper-logo.jpg)

A development blog, titled "The Footy Tipper," provides detailed insights into the progress and findings of this project. You can read the first edition of the blog [here](https://medium.com/@levonrush/the-footy-tipper-a-machine-learning-approach-to-winning-the-pub-tipping-comp-dc07a7325292).

## Workflow
1. **Model Development and EDA**: Developers work within the `research` folder, creating and iterating on Jupyter or Rmarkdown notebooks to develop and refine models.

2. **Commonizing Code**: As models mature, reusable components such as functions and configurations are abstracted and moved to the `pipeline/common` directory. This ensures that both development and production benefit from a single source of truth for these elements.

3. **Model Production**: The refined functions from the `common` directory are then utilized within `pipeline` to structure robust training (`train.py`), inference (`inference.py`) and send (`send.py`) scripts that serve production workflows.

![Workflow Pattern](/images/workflow-pattern.jpg)

## How it Works

**Data Collection and Transformation:** This process starts with the `data-prep.R` script, which is responsible for scraping, data cleaning and feature engineering. Finally, the data is stored in a SQL database to be used by the rest of the pipeline.

**Model Training:**  Following the data preparation, the model development phase is executed in the `train.py`. Python, renowned for its flexibility and extensive suite of machine learning libraries, is used for constructing robust predictive models from the preprocessed data.

**Model Prediction:** It then proceeds to the `inference.py`. Here, the model's predictions are generated.

**Send Predictions:** The `send.py` is responsible for sending the model's predictions to Google Drive the intended recipients. This process is automated and ensures that the predictions are delivered promptly.

Throughout these processes, SQL plays a vital role in data management and transition across various platforms and environments. Docker encapsulates both processes, ensuring portability and facilitating easy deployment.

## Prerequisites

- Docker installed and running on your machine.
- The project secrets:
  - The secrets file (`secrets.env`).
  - Google service account for Google Drive authentication (`service-account-credentials.json`).
- (Optional) R, Python and Visual Studio Code, if you want to develop or debug locally.
- (Optional) Google Cloud Platform account for computation.

## Usage

### Using Docker

1. **Clone this repository.**
    ```bash
    git clone https://github.com/levonrush/footy-tipper.git
    ```

2. **Navigate to the project's directory.**
    ```bash
    cd footy-tipper
    ```

3. **Build the Docker image.** There's no longer a need to specify a `PROCESS` argument since the Dockerfile has been updated to run a series of scripts automatically.
    ```bash
    docker build -t footy-tipper .
    ```

4. **Prepare your environment file and service account token.** Ensure you have a `secrets.env` file and a `service-account-token.json` ready in your project directory but excluded from version control via `.gitignore`.

5. **Run the Docker container.** Replace `<your_host_port>` with the port number you want to use on your host machine (e.g., 4000). Use the `-v` option to securely mount `secrets.env` and `service-account-token.json` into the Docker container.
    ```bash
    docker run -p <your_host_port>:80 \
      -v $(pwd)/secrets.env:/footy-tipper/secrets.env \
      -v $(pwd)/service-account-token.json:/footy-tipper/service-account-token.json \
      footy-tipper
    ```

This sequence ensures that your Docker usage is secure, efficient, and aligns with best practices for handling sensitive information. Remember to keep your `secrets.env` and any sensitive files securely managed and out of version control.


### For Development and Debugging

1. Open the project in your preferred code editor.
2. If needed, set environment variables in a `.env` file or manually in your Python or R session.
3. Run the `data-prep.R` script located in the 'data-prep' folder for data cleaning and feature engineering.
4. For pipeline development, open and execute the `model-training.ipynb` notebook situated in the 'research' folder.
5. If Docker is used, ensure to build and run the Docker image as necessary.

Note: Ensure your Python and R environments have all necessary packages installed to run the scripts and notebooks.

## Contributing

Footy-Tipper welcomes contributions from the community. Please check the issues section of the repository to see how you can contribute.

## Contact

To obtain the project's secrets or for any questions or comments related to this project, please reach out via the repository's issues section.

## Acknowledgements
Special thanks to Seven Seas Hotel for motivating this project, Kate for telling me to make myself a portfolio piece, Victoria and Ernie for the emotional support, and ChatGPT for writing this readme.

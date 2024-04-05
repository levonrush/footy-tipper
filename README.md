# The Footy-Tipper: A Machine Learning Approach to Winning the Pub Tipping Comp

The Footy-Tipper, an open-source Rugby League prediction engine, artfully merges R, Python, and SQL to forecast outcomes of National Rugby League (NRL) matches, creating a holistic data science product. With R for data pipelines and analysis, Python for machine learning modeling, and SQL for data management, the project is a testament to technological synergy. Central to its charm is the incorporation of Reg Reagan's iconic humor and passion, a feature brought to life by ChatGPT's advanced language models. Mimicking Reagan's distinctive style, ChatGPT crafts engaging and witty narratives for sending out predictions, blending accurate, data-driven insights with the beloved cultural fabric of Rugby League. This approach not only elevates the delivery of predictions but also pays homage to the sport's heritage, making The Footy-Tipper a unique intersection of cutting-edge technology and nostalgic homage to Rugby League's golden era, as envisioned through the lens of Reg Reagan's enduring legacy.

![Footy Tipper Logo](/images/footy-tipper-logo.jpg)

A development blog, titled "The Footy Tipper," provides detailed insights into the progress and findings of this project. [You can read the first edition of the blog on Medium](https://medium.com/@levonrush/the-footy-tipper-a-machine-learning-approach-to-winning-the-pub-tipping-comp-dc07a7325292).

## Workflow
1. **Model Development and EDA**: In the `research` folder, Jupyter and Rmarkdown notebooks facilitate exploratory data analysis and initial model prototyping, enabling swift experimentation and model iteration.

2. **Commonizing Code**: Reusable functions and configurations identified during model development are centralized in the `pipeline/common` directory to streamline code management and ensure uniformity across development and production.

3. **Model Production**: Production scripts in the `pipeline` utilize the common functions for efficient model training (`train.py`), prediction (`inference.py`), and result dissemination (`send.py`), automating the end-to-end process from data to delivery.

![Workflow Pattern](/images/workflow.png)

## How it Works

**Data Collection and Transformation:** `data-prep.R` kick-starts the process with data scraping, cleaning, and feature engineering, ensuring that the latest rugby league data is analysis-ready. The data is then stored in a SQL database for downstream processing.

**Model Training:** The `train.py` script takes over to train predictive models using Python's extensive machine learning libraries. It focuses on optimizing model accuracy and performance based on the preprocessed data.

**Model Prediction:** In the `inference.py` script, the trained models generate match predictions, which are then stored in the SQL database, transforming processed data into valuable insights for rugby league followers.

**Send Predictions:** Finally, `send.py` automates the delivery of predictions through Google Drive and emails. The emails are stylized to emulate Reg Reagan's voice, combining predictions with engaging content, thanks to ChatGPT’s linguistic capabilities.

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

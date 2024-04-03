# The Footy-Tipper: A Machine Learning Approach to Winning the Pub Tipping Comp

The Footy-Tipper, an open-source Rugby League prediction engine, artfully merges R, Python, and SQL to forecast outcomes of National Rugby League (NRL) matches, creating a holistic data science product. With R for data pipelines and analysis, Python for machine learning modeling, and SQL for data management, the project is a testament to technological synergy. Central to its charm is the incorporation of Reg Reagan's iconic humor and passion, a feature brought to life by ChatGPT's advanced language models. Mimicking Reagan's distinctive style, ChatGPT crafts engaging and witty narratives for sending out predictions, blending accurate, data-driven insights with the beloved cultural fabric of Rugby League. This approach not only elevates the delivery of predictions but also pays homage to the sport's heritage, making The Footy-Tipper a unique intersection of cutting-edge technology and nostalgic homage to Rugby League's golden era, as envisioned through the lens of Reg Reagan's enduring legacy.

![Footy Tipper Logo](/images/footy-tipper-logo.jpg)

A development blog, titled "The Footy Tipper," provides detailed insights into the progress and findings of this project. You can read the first edition of the blog [here](https://medium.com/@levonrush/the-footy-tipper-a-machine-learning-approach-to-winning-the-pub-tipping-comp-dc07a7325292).

## Workflow
1. **Model Development and EDA (Exploratory Data Analysis)**: The heart of exploratory research and initial model development takes place within the `research` folder. Here, developers and data scientists utilize the interactive environments provided by Jupyter and Rmarkdown notebooks to experiment with various algorithms and analyze data patterns. This sandbox-like setting allows for creative freedom and rapid iteration, which is crucial for discovering innovative solutions and refining predictive models tailored to the dynamic world of rugby league.

2. **Commonizing Code**: As the development phase progresses, certain patterns and functions emerge as common utilities across different models. To harness efficiency and maintain consistency, these reusable components—ranging from data preprocessing functions to complex statistical methods—are migrated to the `pipeline/common` directory. This consolidation process not only reduces redundancy but also ensures that both development and production environments are synchronized, drawing from the same, well-tested codebase for critical operations.

3. **Model Production**: With a robust foundation of common functions in place, the workflow shifts towards operationalizing the models for production. The refined components in the `common` directory serve as the building blocks for constructing systematic scripts that manage the training (`train.py`), inference (`inference.py`), and distribution (`send.py`) processes. These scripts are the workhorses of the production environment, where they perform the heavy lifting of transforming raw data into actionable predictions and delivering those insights directly to the end users. The production workflow embodies the discipline of software engineering, wherein the creative explorations of the research phase are honed into a streamlined, reliable system that operates with precision and scalability.

![Workflow Pattern](/images/workflow-pattern.jpg)

## How it Works

**Data Collection and Transformation:** The journey of data within The Footy-Tipper begins with the `data-prep.R` script. This crucial step involves a meticulous process of data scraping to gather the latest rugby league statistics, followed by a thorough cleaning phase to ensure data quality. Subsequently, the script performs sophisticated feature engineering to uncover deep insights and patterns within the data. The transformed data is then methodically stored in a SQL database, laying a solid foundation for the subsequent stages of the predictive pipeline. By automating the collection and transformation process, the system ensures that the data is not only current but also primed for high-quality predictions.

**Model Training:** After data has been collected and transformed, the next step in The Footy-Tipper's workflow is model training, executed by the `train.py` script. In this phase, Python's versatility shines as it is harnessed to construct powerful predictive models, leveraging its expansive ecosystem of machine learning libraries. The script systematically trains the models on the preprocessed data, fine-tuning and cross-validating to achieve the best possible performance. The end product is a set of rigorously trained models that encapsulate the intricate dynamics of rugby league matches, ready to make their predictions on upcoming games.

**Model Prediction:** Transitioning from training to actual predictions, the `inference.py` script takes center stage. In this script, the previously trained models apply their learned patterns to new data, generating predictions about future match outcomes. These predictions, which are the essence of The Footy-Tipper's purpose, are securely stored within the SQL database. From here, they are ready to be accessed by the sending mechanism, completing the cycle from raw data to actionable insights that inform rugby league enthusiasts and tipsters alike.

**Send Predictions:** The `send.py` script is responsible for distributing the model's predictions. It not only automates the process of sending predictions to Google Drive but also takes charge of emailing the registered subscribers. To add a personal touch and maintain the spirit of the game, the emails are crafted to mimic the style of Reg Reagan, the iconic rugby league personality known for his humor and passion for the sport. This not only keeps the recipients engaged but also strengthens the brand identity of The Footy-Tipper as a source of not only predictions but also entertainment. With the help of ChatGPT's language model fine-tuned to Reg Reagan's linguistic flair, the emails carry a distinctive tone that resonates with rugby fans, making the arrival of predictions something to look forward to each week.

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

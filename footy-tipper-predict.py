from pipeline.utils import run_script

# Define the scripts to run for prediction
scripts = [
    "Rscript pipeline/data-prep.R",
    "python3 pipeline/inference.py",
    "python3 pipeline/send.py"
]

if __name__ == "__main__":
    for script in scripts:
        print(f"Running script: {script}")
        run_script(script)
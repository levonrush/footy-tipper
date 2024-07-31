from pipeline.utils import run_script

# Define the scripts to run for training
scripts = [
    "Rscript pipeline/data-prep.R",
    "python3 pipeline/train.py"
]

if __name__ == "__main__":
    for script in scripts:
        print(f"Running script: {script}")
        run_script(script)
        
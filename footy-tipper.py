import subprocess

# Define the scripts to run in sequence
scripts = [
    "Rscript pipeline/data-prep.R",
    # "python3 pipeline/train.py",
    # "python3 pipeline/inference.py",
    # "python3 pipeline/send.py"
]

def run_script(command):
    """Run the given command as a subprocess and handle its output."""
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully ran {command}")
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error running {command}")
        print(e.stderr.decode())  # Ensure stderr is printed to diagnose issues
        exit(1)

if __name__ == "__main__":
    for script in scripts:
        print(f"Running script: {script}")
        run_script(script)

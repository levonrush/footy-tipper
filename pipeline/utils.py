import subprocess
import time

def run_script(command):
    """Run the given command as a subprocess, handle its output, and time its execution."""
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        end_time = time.time()
        elapsed_time_seconds = end_time - start_time
        elapsed_time_minutes = elapsed_time_seconds / 60
        print(f"Successfully ran {command} in {elapsed_time_seconds:.2f} seconds ({elapsed_time_minutes:.2f} minutes)")
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        elapsed_time_seconds = end_time - start_time
        elapsed_time_minutes = elapsed_time_seconds / 60
        print(f"Error running {command} in {elapsed_time_seconds:.2f} seconds ({elapsed_time_minutes:.2f} minutes)")
        print(e.stderr.decode())  # Ensure stderr is printed to diagnose issues
        exit(1)

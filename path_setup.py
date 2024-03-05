import os
import sys
import pathlib

def setup_path_and_get_project_root():
    # get the directory of the script being run
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # get the parent directory
    parent_dir = os.path.dirname(script_dir)

    # add the parent directory to the system path
    sys.path.insert(0, parent_dir)

    # Get to the root directory
    project_root = pathlib.Path().absolute().parent

    return project_root

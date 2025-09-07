import os
import sys

def add_project_root_to_path():
    # Since the script is located inside core/scripts, need to add project root to path in order to recognize core as a package
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    sys.path.append(project_root)
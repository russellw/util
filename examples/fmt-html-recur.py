import os
import shutil
import subprocess


def beautify_html_files(base_dirs, top_level_dirs):
    """
    Applies `js-beautify` to all HTML files in the specified base directories and their subdirectories.
    Restricts processing of top-level directories to only their immediate files.

    :param base_dirs: List of base directories to search for HTML files recursively.
    :param top_level_dirs: List of directories to process only at the top level.
    """
    # Find the js-beautify executable (handling .bat files explicitly)
    js_beautify_executable = shutil.which("js-beautify")
    if not js_beautify_executable:
        raise FileNotFoundError(
            "js-beautify command not found in PATH. Make sure it is installed and accessible."
        )

    command = [js_beautify_executable, "--end-with-newline", "-r"]

    # Process top-level directories
    for top_level_dir in top_level_dirs:
        for file in os.listdir(top_level_dir):
            file_path = os.path.join(top_level_dir, file)
            if os.path.isfile(file_path) and file.endswith(".html"):
                subprocess.run(command + [file_path], check=True, shell=True)

    # Process other directories recursively
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".html"):
                    file_path = os.path.join(root, file)
                    subprocess.run(command + [file_path], check=True, shell=True)


# List of directories to process recursively
recursive_directories = ["articles"]

# List of directories to process only at the top level
top_level_directories = ["."]

# Run the beautification
beautify_html_files(recursive_directories, top_level_directories)

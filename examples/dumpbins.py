import os
import subprocess

def find_files(directory, extensions):
    """Recursively find files with given extensions in directory."""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extensions):
                yield os.path.join(root, file)

def dumpbin_headers(file_path):
    """Run `dumpbin /headers` on a file and return the output."""
    try:
        result = subprocess.run(["dumpbin", "/headers", file_path], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            print(f"Error processing {file_path}: {result.stderr}")
    except Exception as e:
        print(f"Failed to run dumpbin on {file_path}: {e}")

def extract_machine_line(dumpbin_output):
    """Extract the 'machine' line from dumpbin output."""
    for line in dumpbin_output.split('\n'):
        if 'machine' in line.lower():
            return line.strip()

def main():
    current_directory = os.getcwd()
    target_extensions = ('.exe', '.dll')
    files = find_files(current_directory, target_extensions)

    for file in files:
        output = dumpbin_headers(file)
        if output:
            machine_line = extract_machine_line(output)
            if machine_line:
                print(f"{machine_line} - {file}")

if __name__ == "__main__":
    main()

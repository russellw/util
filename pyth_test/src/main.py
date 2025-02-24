import sys
import os
from pathlib import Path

# Add parent directory to Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Now we can import from utils
from utils.helper import greet, calculate_sum

def main():
    # Test the imported functions
    print(greet("Python Developer"))
    numbers = [1, 2, 3, 4, 5]
    print(f"Sum of numbers {numbers}: {calculate_sum(numbers)}")

if __name__ == "__main__":
    main()
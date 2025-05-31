import time
from datetime import datetime

def print_current_time():
    """Print the current date and time in a readable format"""
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current time: {formatted_time}")

def main():
    print("Starting time display program...")
    print("Press Ctrl+C to stop")
    print("-" * 30)
    
    try:
        while True:
            print_current_time()
            # Sleep for 10 minutes (600 seconds)
            time.sleep(600)
    except KeyboardInterrupt:
        print("\nProgram stopped by user")

if __name__ == "__main__":
    main()
	
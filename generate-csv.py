import csv
import argparse
from faker import Faker

# Initialize the Faker instance for generating fake data
fake = Faker()

def generate_csv(filename, num_records):
    """Generate a CSV file with fake data."""
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['Name', 'Email', 'Phone', 'Address', 'Company', 'Job Title'])

        # Generate fake data records
        for _ in range(num_records):
            writer.writerow([
                fake.name(),
                fake.email(),
                fake.phone_number(),
                fake.address().replace('\n', ', '),
                fake.company(),
                fake.job()
            ])

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate a CSV file with fake data.")
    parser.add_argument('num_records', type=int, help="Number of records to generate.")
    parser.add_argument('output_file', type=str, help="Output CSV file name.")

    args = parser.parse_args()

    # Generate the CSV file
    generate_csv(args.output_file, args.num_records)
    print(f"Generated {args.num_records} records in the file '{args.output_file}'.")

if __name__ == "__main__":
    main()

import csv
from io import StringIO

def dicts_to_csv(dict_list):
    # Get the union of all keys (column headings)
    all_keys = set()
    for d in dict_list:
        all_keys.update(d.keys())
    all_keys = sorted(all_keys)  # Sort keys for consistent ordering

    # Create a StringIO object to hold the CSV data
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=all_keys)

    # Write the header (column headings)
    writer.writeheader()

    # Write each row
    for d in dict_list:
        # Handle lists by joining their elements into a space-separated string
        processed_row = {key: ' '.join(value) if isinstance(value, list) else value
                         for key, value in d.items()}
        writer.writerow(processed_row)

    # Return the CSV content as a string
    return output.getvalue()

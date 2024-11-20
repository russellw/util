import inspect
import csv
from io import StringIO
def dbg(a):
    info = inspect.getframeinfo(inspect.currentframe().f_back)
    print(f"{info.filename}:{info.function}:{info.lineno}: {a}")

def dicts_to_csv(dict_list):
    if not dict_list:
        return ""  # Return empty string if the input list is empty

    # Get the union of all keys (column headings)
    all_keys = set()
    for d in dict_list:
        all_keys.update(d.keys())
    if not all_keys:
        return "\r\n"  # If there are no keys in the dictionaries, return a single line break

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

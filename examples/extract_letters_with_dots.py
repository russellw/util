import regex

def extract_letters_with_dots(input_string):
    # Replace one or more non-letter characters with a single dot
    return regex.sub(r'\P{L}+', '.', input_string)

# Test the function with the provided sample string
sample_string = "Hello, World! 1234"
extracted_letters_with_dots = extract_letters_with_dots(sample_string)
print(extracted_letters_with_dots)

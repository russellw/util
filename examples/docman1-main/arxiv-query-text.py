import requests

# Define the URL for the API request
url = "http://export.arxiv.org/api/query?search_query=all:*&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending"
# Define the URL for the API request with a category filter
url = "http://export.arxiv.org/api/query?search_query=cat:physics*+AND+all:*&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending"
url = (
    "http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=10"
)

# Send the request
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Print the first 500 characters of the response to verify content
    print(response.text)
else:
    print("Failed to fetch data: HTTP Status", response.status_code)

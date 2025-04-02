import xml.etree.ElementTree as ET

import requests

# Define the URL for the API request
url = "http://export.arxiv.org/api/query?search_query=all:*&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending"

# Send the request
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Get the content of the response
    response_content = response.text

    # Parse the XML content
    root = ET.fromstring(response_content)

    # Print titles and URLs of the articles
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text
        link = entry.find('{http://www.w3.org/2005/Atom}link[@rel="alternate"]').attrib[
            "href"
        ]
        print(title)
        print(link)
else:
    print("Failed to fetch data: HTTP Status", response.status_code)

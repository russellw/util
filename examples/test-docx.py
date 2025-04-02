import zipfile
import xml.etree.ElementTree as ET
import sys

# Replace 'yourfile.zip' with the path to your ZIP file
zip_file_path = sys.argv[1]

# Open the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Extract the file's content
            with zip_ref.open('word/document.xml') as xml_file:
                # Parse the XML file
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Print the XML file's contents
                # For demonstration, we print the tag and text of each element
                for elem in root.iter():
                    s=elem.text
                    if s:
                        print(f"Tag: {elem.tag}, Text: {elem.text}")

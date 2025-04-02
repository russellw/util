import etc


# Function to retrieve the icon for a specific document ID
def get_document_icon(document_id):
    # Connecting to the SQLite database
    conn = etc.connect_db()
    cursor = conn.cursor()

    cursor.execute("SELECT icon128 FROM documents WHERE id = ?", (document_id,))
    icon_data = (
        cursor.fetchone()
    )  # fetchone() retrieves one record or None if no records match

    if icon_data:
        return icon_data[0]  # Returns the blob data of the icon
    else:
        return None  # Returns None if the document does not exist


# Usage example
document_id = 1
icon_bytes = get_document_icon(document_id)
if icon_bytes:
    print("Icon retrieved successfully.")
    print(len(icon_bytes))
    # Here you can write code to save or process the icon bytes further
    f = open("/t/a.png", "wb")
    f.write(icon_bytes)
else:
    print("No icon found for the given document ID or error occurred.")

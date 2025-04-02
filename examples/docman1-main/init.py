import os

import etc

with etc.conn:
    cursor = etc.conn.cursor()
    try:
        for root, dirs, files in os.walk(r"C:\doc\computing"):
            for file in files:
                if file.endswith(".pdf"):
                    file = os.path.join(root, file)
                    print(file)
                    etc.import_file(file, cursor)
                    break
    finally:
        cursor.close()

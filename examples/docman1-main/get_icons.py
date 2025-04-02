from dataclasses import dataclass

import etc


@dataclass
class Document:
    ID: int
    name: str
    icon: bytes


conn = etc.connect_db()
cursor = conn.cursor()
cursor.execute("SELECT id, name, icon128 FROM documents")
rows = cursor.fetchall()
documents = [Document(ID=row[0], name=row[1], icon=row[2]) for row in rows]

for doc in documents:
    f = open(f"/t/icons/{doc.name}.png", "wb")
    f.write(doc.icon)

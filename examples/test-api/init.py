import os

import etc

file = "/t/a.txt"
name = "test file"
text = open(file).read()
icon = etc.image_bytes(etc.text_icon(text))
raw_data = open(file, "rb").read()

os.remove(etc.db_path)
etc.init_db()
conn = etc.connect_db()
cursor = conn.cursor()
cursor.execute(
    "insert into documents(file,name,icon256,text,raw_data) values(?,?,?,?,?)",
    (file, name, icon, text, raw_data),
)
conn.commit()

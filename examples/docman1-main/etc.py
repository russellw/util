import os
import platform
import sqlite3
import sys
import textwrap
from io import BytesIO

import fitz
from PIL import Image, ImageDraw, ImageFont, ImageTk


class Document:
    @property
    def photo(self):
        if not hasattr(self, "_photo"):
            image = Image.open(BytesIO(self.icon))
            self._photo = ImageTk.PhotoImage(image)
        return self._photo

    def __repr__(self):
        if hasattr(self, "ID"):
            return f"#{self.ID}"
        return super().__repr__()


def get_resource_path(relative_path):
    """Get the absolute path to the resource, works for dev and PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def norm(file):
    if file and platform.system() == "Windows":
        return file.replace("/", "\\")
    return file


def pdf_icon(file):
    doc = fitz.open(file)
    page = doc[0]

    # Original dimensions of the page
    original_width, original_height = page.rect.width, page.rect.height

    # Calculate scaling factors for width and height
    scale_width = icon_size / original_width
    scale_height = icon_size / original_height

    # Use the smaller scaling factor to maintain aspect ratio
    scale = min(scale_width, scale_height)

    # Create a transformation matrix for the pixmap
    matrix = fitz.Matrix(scale, scale)

    # Generate the pixmap
    pix = page.get_pixmap(matrix=matrix)
    return pix.tobytes()


def pdf_text(file):
    doc = fitz.open(file)
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\f".join(text)


def text_icon(text, font_name="arial", font_size=18):
    image = Image.new("RGB", (icon_size, icon_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(font_name, font_size)
    except IOError:
        print(f"Could not load font {font_name}, using default font.")
        font = ImageFont.load_default()
    text = textwrap.fill(text, width=72)
    draw.text((5, 5), text, fill=(0, 0, 0), font=font)
    return image_bytes(image)


def image_bytes(image):
    b = BytesIO()
    image.save(b, "PNG")
    return b.getvalue()


def import_file(file, cursor):
    doc = Document()
    doc.file = file
    doc.name = os.path.splitext(os.path.basename(file))[0]
    doc.icon = pdf_icon(file)
    doc.text = pdf_text(file)
    doc.raw_data = open(file, "rb").read()
    cursor.execute(
        "insert into documents(file,name,icon256,text,raw_data,folder) values(?,?,?,?,?,1)",
        (doc.file, doc.name, doc.icon, doc.text, doc.raw_data),
    )
    doc.ID = cursor.lastrowid
    return doc


icon_size = 256

# File path
user_profile = os.environ["USERPROFILE"]
documents_dir = os.path.join(user_profile, "Documents")
db_path = os.path.join(documents_dir, "lumagraph.db")

# Connect to database
conn = sqlite3.connect(db_path)
conn.execute("PRAGMA foreign_keys=ON")

# Initialize tables
with conn:
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        if len(tables) == 0:
            cursor.execute(
                """
            CREATE TABLE folders (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                added TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            ) STRICT;
            """
            )

            cursor.execute(
                """
            INSERT INTO folders(name) VALUES('Inbox');
            """
            )

            cursor.execute(
                """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                file TEXT,
                text TEXT,
                icon256 BLOB,
                raw_data BLOB,
                folder INTEGER NOT NULL,
                added TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (folder) REFERENCES folders (id)
            ) STRICT;
            """
            )
    finally:
        cursor.close()

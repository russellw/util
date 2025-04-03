import sqlite3
import textwrap
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

db_path = r"C:\Users\User\Documents\lumagraph.db"


def connect_db():
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    if len(tables) == 0:
        print("No tables found. Initializing database.")
        cursor.execute(open("schema.sql").read())


def text_icon(text, size=256, font_name="arial", font_size=18):
    image = Image.new("RGB", (size, size), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(font_name, font_size)
    except IOError:
        print(f"Could not load font {font_name}, using default font.")
        font = ImageFont.load_default()
    text = textwrap.fill(text, width=72)
    draw.text((5, 5), text, fill=(0, 0, 0), font=font)
    return image


def image_bytes(image):
    b = BytesIO()
    image.save(b, "PNG")
    return b.getvalue()

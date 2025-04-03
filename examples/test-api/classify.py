import os
import random

from openai import OpenAI


def find_txt_files(directory):
    txt_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                full_path = os.path.join(root, file)
                txt_files.append((full_path, os.path.basename(root)))
    return txt_files


directory = r"C:\w\newsgroup documents"
file_categories = find_txt_files(directory)
random.shuffle(file_categories)
file_categories = file_categories[:1]
categories = sorted(list(set([a[1] for a in file_categories])))
print(categories)

client = OpenAI()


def ask(system_prompt, user_prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


system_prompt = (
    "Your job is to classify documents into one of the following categories: "
    + ", ".join(categories)
)
for file, category in file_categories:
    user_prompt = "Classify this document:\n\n" + open(file).read()
    print(user_prompt)
    print(category)
    r = ask(system_prompt, user_prompt)
    print(r)

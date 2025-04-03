import mailbox

from openai import OpenAI

# Specify the path to your mbox file
mbox_file_path = r"C:\doc\data\devel@lists.fedoraproject.org-2024-04-05-2024-05-07.mbox"

# Open the mbox file
mbox = mailbox.mbox(mbox_file_path)

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


system_prompt = "You will be given the full text of a mailing list message, including headers. Your job is to figure out whether it constitutes an action item."

# Initialize a counter
count = 0

# Iterate through all messages in the mbox
for message in mbox:
    s = message.as_string()
    user_prompt = "Message text:\n\n" + s
    r = ask(system_prompt, user_prompt)
    print(r)
    print("-" * 80)  # Print a line for better separation between messages

    # Increment the counter
    count += 1

    # Break the loop after printing 10 messages
    if count == 3:
        break

import mailbox


def print_first_10_messages(mbox_file_path):
    # Open the mbox file
    mbox = mailbox.mbox(mbox_file_path)

    # Initialize a counter
    count = 0

    # Iterate through all messages in the mbox
    for message in mbox:
        # Print the entire message as a raw string
        print(message.as_string())
        print("-" * 80)  # Print a line for better separation between messages

        # Increment the counter
        count += 1

        # Break the loop after printing 10 messages
        if count == 10:
            break


# Specify the path to your mbox file
mbox_file_path = r"C:\doc\data\devel@lists.fedoraproject.org-2024-04-05-2024-05-07.mbox"

# Call the function to print the first 10 messages
print_first_10_messages(mbox_file_path)

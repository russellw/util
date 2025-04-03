import mailbox


def print_full_messages(mbox_file_path):
    # Open the mbox file
    mbox = mailbox.mbox(mbox_file_path)

    # Iterate through all messages in the mbox
    for message in mbox[:3]:
        # Print the entire message as a raw string
        print(message.as_string())
        print("-" * 80)  # Print a line for better separation between messages


# Specify the path to your mbox file
mbox_file_path = r"C:\doc\data\devel@lists.fedoraproject.org-2024-04-05-2024-05-07.mbox"

# Call the function to print full messages
print_full_messages(mbox_file_path)

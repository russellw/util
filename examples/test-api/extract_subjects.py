import mailbox


def extract_subjects(mbox_file_path):
    # Open the mbox file
    mbox = mailbox.mbox(mbox_file_path)

    # Iterate through all messages in the mbox
    for message in mbox:
        # Get the subject of the message
        subject = message["subject"]
        print(subject)


# Specify the path to your mbox file
mbox_file_path = r"C:\doc\data\devel@lists.fedoraproject.org-2024-04-05-2024-05-07.mbox"

# Call the function to extract and print subjects
extract_subjects(mbox_file_path)

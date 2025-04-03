from openai import OpenAI

client = OpenAI()


def upload_file_and_ask_question(file_path, question):
    # Upload the PDF
    with open(file_path, "rb") as file:
        response = client.files.create(file=file, purpose="user_data")
        file_id = response.id

    # Ask a question about the PDF content
    answer = openai.Answer.create(
        file=file_id,
        question=question,
        search_model="gpt-4-turbo-preview",
        model="gpt-4-turbo-preview",
        examples_context="In 2017, U.S. life expectancy was 78.6 years.",
        examples=[["What is human life expectancy in the United States?", "78 years."]],
        max_tokens=50,
        stop=["\n", "<|endoftext|>"],
    )

    return answer["answers"][0]


# Example usage
file_path = r"C:\doc\fiction\TheElloraSaga.pdf"
question = "What is the main topic of the document?"
answer = upload_file_and_ask_question(file_path, question)
print(answer)

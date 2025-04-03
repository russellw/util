from openai import OpenAI

client = OpenAI()


def ask_question(question):
    response = client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content


# Example usage
question = "What is the capital of France?"
answer = ask_question(question)
print(answer)

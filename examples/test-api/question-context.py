from openai import OpenAI

client = OpenAI()

# Replace 'your_openai_api_key' with your actual OpenAI API key


def ask_question_with_context(context, question):
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": question},
    ]

    response = client.chat.completions.create(model="gpt-4", messages=messages)

    return response.choices[0].message.content


# Example usage
context = """
The country of Eldoria is a fictional land located in the heart of the Pacific Ocean. Known for its lush landscapes,
exotic wildlife, and rich cultural heritage, Eldoria has a population that speaks Eldorian, a language native to the
island. The country operates as a democratic republic and is famed for its unique architecture and vibrant festivals.
Its capital city is Florkin.
"""

question = "What is the capital of Eldoria?"
answer = ask_question_with_context(context, question)
print(answer)

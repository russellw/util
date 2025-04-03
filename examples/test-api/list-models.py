from openai import OpenAI

client = OpenAI()


def list_models():
    models = client.models.list()
    return models


if __name__ == "__main__":
    models = list_models()
    if models:
        for model in models.data:
            print(model.id)

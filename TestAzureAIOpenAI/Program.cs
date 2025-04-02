using Azure.AI.OpenAI;

namespace TestAzureAIOpenAI {
internal class Program {
	static void Main(string[] args) {
		var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? throw new Exception("OPENAI_API_KEY not set");
		OpenAIClient client = new(apiKey);
		RunAsync(client).GetAwaiter().GetResult();
	}

	static async Task RunAsync(OpenAIClient client) {
		var chatCompletionsOptions = new ChatCompletionsOptions() {
			DeploymentName = "gpt-3.5-turbo",
			Messages =
				{
					new ChatRequestSystemMessage("You are a helpful assistant. You will talk like a pirate."),
					new ChatRequestUserMessage("Can you help me?"),
					new ChatRequestAssistantMessage("Arrrr! Of course, me hearty! What can I do for ye?"),
					new ChatRequestUserMessage("What's the best way to train a parrot?"),
				},
		};
		await foreach (StreamingChatCompletionsUpdate chatUpdate in client.GetChatCompletionsStreaming(chatCompletionsOptions)) {
			if (chatUpdate.Role.HasValue)
				Console.Write($"{chatUpdate.Role.Value.ToString().ToUpperInvariant()}: ");
			if (!string.IsNullOrEmpty(chatUpdate.ContentUpdate))
				Console.Write(chatUpdate.ContentUpdate);
		}
	}
}
}

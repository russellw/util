using System;
using System.Collections.Immutable;
using System.IO;
using MimeDetective;
using MimeDetective.Engine;

class Program {
	// https://github.com/MediatedCommunications/Mime-Detective
	static readonly ContentInspector Inspector =
		new ContentInspectorBuilder() {
			Definitions =
				new MimeDetective.Definitions
					.ExhaustiveBuilder() { UsageType = MimeDetective.Definitions.Licensing.UsageType.PersonalNonCommercial }
					.Build()
		}
			.Build();

	static void Main(string[] args) {
		string rootDirectory = ".";
		if (args.Length > 0)
			rootDirectory = args[0];
		CheckDirectory(rootDirectory);
	}

	static void CheckDirectory(string directoryPath) {
		try {
			string[] files = Directory.GetFiles(directoryPath);
			foreach (string file in files) {
				CheckFile(file);
			}

			string[] subDirectories = Directory.GetDirectories(directoryPath);
			foreach (string subDirectory in subDirectories) {
				CheckDirectory(subDirectory);
			}
		} catch (Exception ex) {
			Console.WriteLine($"Error processing {directoryPath}: {ex.Message}");
		}
	}

	static void CheckFile(string filePath) {
		string extension = Path.GetExtension(filePath).ToLower();
		if (string.IsNullOrEmpty(extension)) {
			return; // Skip files without extensions
		}
		extension = extension[1..];

		var likelyExtensions = new List<string>();
		var Results = Inspector.Inspect(filePath);
		foreach (var Result in Results)
			foreach (var e in Result.Definition.File.Extensions) {
				likelyExtensions.Add(e);
			}

		if (!likelyExtensions.Any())
			return;
		if (likelyExtensions.Contains(extension))
			return;
		switch (extension) {
		case "dll":
			if (likelyExtensions.Contains("exe"))
				return;
			break;
		case "log":
			if (likelyExtensions.Contains("txt"))
				return;
			break;
		case "htm":
			if (likelyExtensions.Contains("html"))
				return;
			break;
		}

		Console.WriteLine(filePath);
		foreach (var Result in Results) {
			Console.Write('\t');
			Console.WriteLine(Result.Definition.File.Description);

			Console.Write("\t\t");
			Console.WriteLine(string.Join(", ", Result.Definition.File.Extensions));

			Console.Write("\t\t");
			Console.WriteLine(string.Join(", ", Result.Definition.File.Categories));

			Console.Write("\t\t");
			Console.WriteLine(Result.Definition.File.MimeType);
		}
	}
}

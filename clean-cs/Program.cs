using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

static class Program {
	static bool inplace;

	static void Descend(string path) {
		foreach (var entry in new DirectoryInfo(path).EnumerateFileSystemInfos()) {
			if (entry is DirectoryInfo) {
				if (!entry.Name.StartsWith('.'))
					Descend(entry.FullName);
				continue;
			}
			if (".cs" == entry.Extension)
				Do(entry.FullName);
		}
	}

	static void Do(string file) {
		var text = File.ReadAllText(file);
		var old = text;
		var tree = CSharpSyntaxTree.ParseText(text, CSharpParseOptions.Default, file);
		if (tree.GetDiagnostics().Any()) {
			foreach (var diagnostic in tree.GetDiagnostics())
				Console.Error.WriteLine(diagnostic);
			Environment.Exit(1);
		}
		SyntaxNode root = tree.GetCompilationUnitRoot();

		// Apply transformations
		root = new CapitalizeComments(root).Visit(root);
		root = new RemoveRedundantBraces().Visit(root);
		root = new SortCaseLabels().Visit(root);
		root = new SortCaseSections().Visit(root);
		root = new SortComparison().Visit(root);
		root = new SortMembers().Visit(root);

		tree = CSharpSyntaxTree.Create((CSharpSyntaxNode)root);
		text = tree.ToString();
		if (!text.EndsWith('\n'))
			text += '\n';
		if (inplace) {
			if (old == text)
				return;
			WriteText(file, text);
			Console.Error.WriteLine(file);
			return;
		}
		Console.Write(text);
	}

	static void Help() {
		var name = typeof(Program).Assembly.GetName().Name;
		Console.WriteLine($"Usage: {name} [options] path...");
		Console.WriteLine("");
		Console.WriteLine("-h  Show help");
		Console.WriteLine("-V  Show version");
		Console.WriteLine("-i  In-place edit");
		Console.WriteLine("-r  Recur into directories");
	}

	static void Main(string[] args) {
		var options = true;
		var recursive = false;
		var paths = new List<string>();
		foreach (var arg in args) {
			var s = arg;
			if (options) {
				if ("--" == s) {
					options = false;
					continue;
				}
				if (s.StartsWith('-')) {
					while (s.StartsWith('-'))
						s = s[1..];
					switch (s) {
					case "?":
					case "h":
					case "help":
						Help();
						return;
					case "V":
					case "v":
					case "version":
						Version();
						return;
					case "i":
						inplace = true;
						break;
					case "r":
					case "recursive":
						recursive = true;
						break;
					default:
						Console.Error.WriteLine("{0}: unknown option", arg);
						Environment.Exit(1);
						break;
					}
					continue;
				}
			}
			paths.Add(s);
		}
		if (0 == paths.Count) {
			Help();
			return;
		}

		foreach (var path in paths)
			if (Directory.Exists(path)) {
				if (!recursive) {
					Console.Error.WriteLine(path + " is a directory, use -r to recur on all .cs files therein");
					Environment.Exit(1);
				}
				Descend(path);
			} else
				Do(path);
	}

	static void Version() {
		var name = typeof(Program).Assembly.GetName().Name;
		var version = typeof(Program).Assembly.GetName()?.Version?.ToString(2);
		Console.WriteLine($"{name} {version}");
	}

	static void WriteText(string file, string text) {
		using var writer = new StreamWriter(file);
		writer.NewLine = "\n";
		writer.Write(text);
	}
}

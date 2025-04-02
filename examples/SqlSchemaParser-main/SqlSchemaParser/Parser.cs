using System.Diagnostics;
using System.Text;

namespace SqlSchemaParser;
public sealed class Parser {
	public static void Parse(string file, string text, Schema schema) {
		_ = new Parser(file, text, schema);
	}

	delegate bool Callback();

	const int kDoublePipe = -2;
	const int kGreaterEqual = -3;
	const int kLessEqual = -4;
	const int kNotEqual = -5;
	const int kNumber = -6;
	const int kQuotedName = -7;
	const int kStringLiteral = -8;
	const int kWord = -9;

	readonly string file;
	readonly string text;
	readonly Schema schema;

	int textIndex;
	readonly List<Token> tokens = new();

	int tokenIndex = -1;
	readonly List<int> ignored = new();

	Parser(string file, string text, Schema schema) {
		if (!text.EndsWith('\n'))
			text += '\n';
		this.file = file;
		this.text = text;
		this.schema = schema;
		Lex();
		Debug.Assert(textIndex == text.Length);
		tokenIndex = 0;
		while (tokens[tokenIndex].Type != -1) {
			var token = tokens[tokenIndex];
			var location = new Location(file, text, token.Start);
			switch (Word()) {
			case "create":
				switch (Word(1)) {
				case "table": {
					tokenIndex += 2;
					var table = new Table(UnqualifiedName());
					schema.Add(location, table);
					while (!Eat('('))
						Ignore();
					do {
						ColumnOrTableConstraint(table, IsElementEnd);
						ToElementEnd();
					} while (Eat(','));
					Expect(')');
					StatementEnd();
					continue;
				}
				}
				break;
			case "alter":
				switch (Word(1)) {
				case "table": {
					tokenIndex += 2;
#pragma warning disable CS0219 // Variable is assigned but its value is never used
					var ifExists = false;
#pragma warning restore CS0219 // Variable is assigned but its value is never used
					if (Word() == "if" && Word(1) == "exists") {
						tokenIndex += 2;
						ifExists = true;
					}
					Eat("only");
					var table = schema.GetTable(location, UnqualifiedName());
					switch (Word()) {
					case "add": {
						tokenIndex++;
						ColumnOrTableConstraint(table, IsStatementEnd);
						StatementEnd();
						continue;
					}
					}
					break;
				}
				}
				break;
			}
			Ignore();
		}
		for (int i = 0; i < ignored.Count;) {
			var t = ignored[i++];
			var token = tokens[t];
			var location = new Location(file, text, token.Start);
			var end = token.End;
			while (i < ignored.Count && ignored[i] == t + 1) {
				t = ignored[i++];
				token = tokens[t];
				end = token.End;
			}
			var span = new Span(location, end);
			schema.Ignored.Add(span);
		}
	}

	bool IsElementEnd() {
		var token = tokens[tokenIndex];
		switch (token.Type) {
		case ')':
		case ',':
			return true;
		}
		return false;
	}

	bool IsStatementEnd() {
		var token = tokens[tokenIndex];
		switch (token.Type) {
		case -1:
		case ';':
			return true;
		case kWord:
			switch (Word()) {
			case "go":
			case "create":
			case "alter":
				return true;
			}
			break;
		}
		return false;
	}

	string UnqualifiedName() {
		string name;
		do
			name = Name();
		while (Eat('.'));
		return name;
	}

	void ToElementEnd() {
		while (!IsElementEnd())
			Ignore();
	}

	void StatementEnd() {
		Eat(';');
		Eat("go");
	}

	void Expect(char k) {
		if (!Eat(k))
			throw Error("expected " + k);
	}

	void Expect(string s) {
		if (!Eat(s))
			throw Error("expected " + s.ToUpperInvariant());
	}

	string DataTypeName() {
		switch (Word()) {
		case "character":
		case "char":
			switch (Word(1)) {
			case "large":
				switch (Word(2)) {
				case "object":
					tokenIndex += 3;
					return "clob";
				}
				break;
			case "varying":
				tokenIndex += 2;
				return "varchar";
			}
			break;
		case "binary":
			switch (Word(1)) {
			case "large":
				switch (Word(2)) {
				case "object":
					tokenIndex += 3;
					return "blob";
				}
				break;
			}
			break;
		case "double":
			switch (Word(1)) {
			case "precision":
				tokenIndex += 2;
				return "double";
			}
			break;
		case "long":
			switch (Word(1)) {
			case "raw":
			case "varbinary":
			case "varchar": {
				var name = "long " + Word(1);
				tokenIndex += 2;
				return name;
			}
			}
			break;
		case "time":
			switch (Word(1)) {
			case "with":
				switch (Word(2)) {
				case "timezone":
					tokenIndex += 3;
					return "time with timezone";
				}
				break;
			}
			break;
		case "timestamp":
			switch (Word(1)) {
			case "with":
				switch (Word(2)) {
				case "timezone":
					tokenIndex += 3;
					return "timestamp with timezone";
				}
				break;
			}
			break;
		case "interval":
			switch (Word(1)) {
			case "day":
				switch (Word(2)) {
				case "to":
					switch (Word(3)) {
					case "second":
						tokenIndex += 4;
						return "interval day to second";
					}
					break;
				}
				break;
			case "year":
				switch (Word(2)) {
				case "to":
					switch (Word(3)) {
					case "month":
						tokenIndex += 4;
						return "interval year to month";
					}
					break;
				}
				break;
			}
			break;
		}
		return Name();
	}

	DataType DataType() {
		var a = new DataType(DataTypeName());
		if (Eat('(')) {
			if (a.TypeName == "enum") {
				a.Values = new();
				do
					a.Values.Add(StringLiteral());
				while (Eat(','));
			} else {
				a.Size = Int();
				if (Eat(','))
					a.Scale = Int();
			}
			Expect(')');
		}
		return a;
	}

	int Int() {
		var token = tokens[tokenIndex];
		if (token.Type != kNumber)
			throw Error("expected integer");
		tokenIndex++;
		return int.Parse(token.Value!, System.Globalization.CultureInfo.InvariantCulture);
	}

	Action GetAction() {
		switch (Word()) {
		case "cascade":
			tokenIndex++;
			return Action.CASCADE;
		case "no":
			tokenIndex++;
			Expect("action");
			return Action.NO_ACTION;
		case "restrict":
			tokenIndex++;
			return Action.NO_ACTION;
		case "set":
			tokenIndex++;
			switch (Word()) {
			case "null":
				tokenIndex++;
				return Action.SET_NULL;
			case "default":
				tokenIndex++;
				return Action.SET_DEFAULT;
			}
			throw Error("expected replacement value");
		}
		throw Error("expected action");
	}

	void ForeignKey(Table table, Column? column, Callback isEnd) {
		if (Eat("foreign"))
			Expect("key");
		var key = new ForeignKey();

		// Columns
		if (column == null) {
			Expect('(');
			do
				key.Columns.Add(Column(table));
			while (Eat(','));
			Expect(')');
		} else
			key.Columns.Add(column);

		// References
		Expect("references");
		key.RefTable = Table();
		if (Eat('(')) {
			do
				key.RefColumns.Add(Column(key.RefTable));
			while (Eat(','));
			Expect(')');
		} else {
			if (key.RefTable.PrimaryKey == null)
				throw Error($"{key.RefTable} does not have a primary key", false);
			key.RefColumns.AddRange(key.RefTable.PrimaryKey.Columns);
		}

		table.ForeignKeys.Add(key);

		// Search the postscript for actions
		while (!isEnd()) {
			switch (Word()) {
			case "on":
				switch (Word(1)) {
				case "delete":
					tokenIndex += 2;
					key.OnDelete = GetAction();
					continue;
				case "update":
					tokenIndex += 2;
					key.OnUpdate = GetAction();
					continue;
				}
				break;
			}
			Ignore();
		}
	}

	Key Key(Table table, Column? column) {
		switch (Word()) {
		case "primary":
			tokenIndex++;
			Expect("key");
			break;
		case "unique":
			tokenIndex++;
			Eat("key");
			break;
		case "key":
			tokenIndex++;
			break;
		default:
			throw Error("expected key");
		}
		var key = new Key();

		if (column == null) {
			while (!Eat('('))
				Ignore();
			do
				key.Add(Column(table));
			while (Eat(','));
			Expect(')');
		} else
			key.Add(column);
		return key;
	}

	Column Column(Table table) {
		var token = tokens[tokenIndex];
		var location = new Location(file, text, token.Start);
		return table.GetColumn(location, Name());
	}

	Table Table() {
		var token = tokens[tokenIndex];
		var location = new Location(file, text, token.Start);
		return schema.GetTable(location, UnqualifiedName());
	}

	void TableConstraint(Table table, Callback isEnd) {
		var token = tokens[tokenIndex];
		var location = new Location(file, text, token.Start);
		switch (Word()) {
		case "foreign":
			ForeignKey(table, null, isEnd);
			return;
		case "primary":
			table.AddPrimaryKey(location, Key(table, null));
			return;
		case "unique":
		case "key":
			table.UniqueKeys.Add(Key(table, null));
			return;
		}
	}

	void ColumnOrTableConstraint(Table table, Callback isEnd) {
		// Might be a table constraint
		if (Eat("constraint")) {
			Name();
			TableConstraint(table, isEnd);
			return;
		}
		switch (Word()) {
		case "foreign":
		case "key":
		case "primary":
		case "unique":
		case "check":
		case "exclude":
			TableConstraint(table, isEnd);
			return;
		}

		// This is a column
		var token = tokens[tokenIndex];
		var location = new Location(file, text, token.Start);
		var column = new Column(Name(), DataType());
		table.Add(location, column);

		// Search the postscript for column constraints
		while (!isEnd()) {
			switch (Word()) {
			case "foreign":
			case "references":
				ForeignKey(table, column, isEnd);
				continue;
			case "primary":
				table.AddPrimaryKey(location, Key(table, column));
				continue;
			case "null":
				tokenIndex++;
				continue;
			case "not":
				switch (Word(1)) {
				case "null":
					tokenIndex += 2;
					column.Nullable = false;
					continue;
				}
				break;
			}
			Ignore();
		}
	}

	void Ignore() {
		var i = tokenIndex;
		int depth = 0;
		do {
			var token = tokens[i];
			switch (token.Type) {
			case -1:
				throw Error(depth == 0 ? "unexpected end of file" : "unclosed (");
			case '(':
				depth++;
				break;
			case ')':
				depth--;
				break;
			}
			ignored.Add(i++);
		} while (depth > 0);
		tokenIndex = i;
	}

	string StringLiteral() {
		var token = tokens[tokenIndex];
		if (token.Type != kStringLiteral)
			throw Error("expected string literal");
		tokenIndex++;
		return token.Value!;
	}

	string Name() {
		var token = tokens[tokenIndex];
		switch (token.Type) {
		case kWord:
		case kQuotedName:
			tokenIndex++;
			return token.Value!;
		}
		throw Error("expected name");
	}

	QualifiedName QualifiedName() {
		var a = new QualifiedName();
		do
			a.Names.Add(Name());
		while (Eat('.'));
		return a;
	}

	bool Eat(int k) {
		var token = tokens[tokenIndex];
		if (token.Type == k) {
			tokenIndex++;
			return true;
		}
		return false;
	}

	bool Eat(string s) {
		var token = tokens[tokenIndex];
		if (token.Type == kWord && token.Value == s) {
			tokenIndex++;
			return true;
		}
		return false;
	}

	string? Word(int i = 0) {
		var token = tokens[tokenIndex + i];
		if (token.Type == kWord)
			return token.Value!;
		return null;
	}

	void Lex() {
		Debug.Assert(textIndex == 0);
		while (textIndex < text.Length) {
			int k = text[textIndex];
			var i = textIndex + 1;
			switch (k) {
			case '|':
				switch (text[i]) {
				case '|':
					i = textIndex + 2;
					k = kDoublePipe;
					break;
				}
				break;
			case '!':
				switch (k) {
				case '=':
					i = textIndex + 2;
					k = kNotEqual;
					break;
				case '<':
					// https://stackoverflow.com/questions/77475517/what-are-the-t-sql-and-operators-for
					i = textIndex + 2;
					k = kGreaterEqual;
					break;
				case '>':
					i = textIndex + 2;
					k = kLessEqual;
					break;
				}
				break;
			case '>':
				switch (k) {
				case '=':
					i = textIndex + 2;
					k = kGreaterEqual;
					break;
				}
				break;
			case '<':
				switch (k) {
				case '=':
					i = textIndex + 2;
					k = kLessEqual;
					break;
				case '>':
					i = textIndex + 2;
					k = kNotEqual;
					break;
				}
				break;
			case '-':
				switch (text[i]) {
				case '-':
					textIndex = text.IndexOf('\n', i);
					continue;
				}
				break;
			case '#':
				textIndex = text.IndexOf('\n', i);
				continue;
			case '/':
				switch (text[i]) {
				case '*':
					i = text.IndexOf("*/", textIndex + 2);
					if (i < 0)
						throw Error("unclosed /*");
					textIndex = i + 2;
					continue;
				}
				break;
			case ',':
			case '=':
			case '&':
			case ';':
			case '+':
			case '%':
			case '(':
			case ')':
			case '~':
			case '*':
			case '@':
				break;
			case '\n':
			case '\r':
			case '\t':
			case '\f':
			case '\v':
			case ' ':
				textIndex = i;
				continue;
			case 'N':
				switch (text[i]) {
				case '\'':
					// We are reading everything as Unicode anyway
					// so the prefix has no special meaning
					textIndex = i;
					SingleQuote();
					continue;
				}
				LexWord();
				continue;
			case 'A':
			case 'B':
			case 'C':
			case 'D':
			case 'E':
			case 'F':
			case 'G':
			case 'H':
			case 'I':
			case 'J':
			case 'K':
			case 'L':
			case 'M':
			case 'O':
			case 'P':
			case 'Q':
			case 'R':
			case 'S':
			case 'T':
			case 'U':
			case 'V':
			case 'W':
			case 'X':
			case 'Y':
			case 'Z':
			case '_':
			case 'a':
			case 'b':
			case 'c':
			case 'd':
			case 'e':
			case 'f':
			case 'g':
			case 'h':
			case 'i':
			case 'j':
			case 'k':
			case 'l':
			case 'm':
			case 'n':
			case 'o':
			case 'p':
			case 'q':
			case 'r':
			case 's':
			case 't':
			case 'u':
			case 'v':
			case 'w':
			case 'x':
			case 'y':
			case 'z':
				LexWord();
				continue;
			case '0':
			case '1':
			case '2':
			case '3':
			case '4':
			case '5':
			case '6':
			case '7':
			case '8':
			case '9':
				LexNumber();
				continue;
			case '.':
				if (char.IsDigit(text[i])) {
					LexNumber();
					continue;
				}
				break;
			case '$':
				if (char.IsDigit(text[i])) {
					textIndex = i;
					LexNumber();
					continue;
				}
				throw Error("stray " + (char)k);
			case '"':
				DoubleQuote();
				continue;
			case '\'':
				SingleQuote();
				continue;
			case '`':
				Backquote();
				continue;
			case '[':
				Square();
				continue;
			default:
				// Common letters are handled in the switch for speed
				// but there are other letters in Unicode
				if (char.IsLetter((char)k)) {
					LexWord();
					continue;
				}

				// Likewise digits
				if (char.IsDigit((char)k)) {
					LexWord();
					continue;
				}

				// And whitespace
				if (char.IsWhiteSpace((char)k)) {
					textIndex = i;
					continue;
				}

				throw Error("stray " + (char)k);
			}
			tokens.Add(new Token(textIndex, i, k));
			textIndex = i;
		}
		tokens.Add(new Token(textIndex, textIndex, -1));
	}

	// For string literals, single quote is reliably portable across dialects
	void SingleQuote() {
		Debug.Assert(text[textIndex] == '\'');
		var i = textIndex + 1;
		var sb = new StringBuilder();
		while (i < text.Length) {
			switch (text[i]) {
			case '\\':
				switch (text[i + 1]) {
				case '\'':
				case '\\':
					i++;
					break;
				}
				break;
			case '\'':
				i++;
				switch (text[i]) {
				case '\'':
					break;
				default:
					tokens.Add(new Token(textIndex, i, kStringLiteral, sb.ToString()));
					textIndex = i;
					return;
				}
				break;
			}
			sb.Append(text[i++]);
		}
		throw Error("unclosed '");
	}

	// For unusual identifiers, MySQL uses backquotes
	void Backquote() {
		Debug.Assert(text[textIndex] == '`');
		var i = textIndex + 1;
		var sb = new StringBuilder();
		while (i < text.Length) {
			switch (text[i]) {
			case '\\':
				switch (text[i + 1]) {
				case '`':
				case '\\':
					i++;
					break;
				}
				break;
			case '`':
				i++;
				switch (text[i]) {
				case '`':
					break;
				default:
					tokens.Add(new Token(textIndex, i, kQuotedName, sb.ToString()));
					textIndex = i;
					return;
				}
				break;
			}
			sb.Append(text[i++]);
		}
		throw Error("unclosed `");
	}

	// For unusual identifiers, SQL Server uses square brackets
	void Square() {
		Debug.Assert(text[textIndex] == '[');
		var i = textIndex + 1;
		var sb = new StringBuilder();
		while (i < text.Length) {
			switch (text[i]) {
			case ']':
				i++;
				switch (text[i]) {
				case ']':
					break;
				default:
					tokens.Add(new Token(textIndex, i, kQuotedName, sb.ToString().ToLowerInvariant()));
					textIndex = i;
					return;
				}
				break;
			}
			sb.Append(text[i++]);
		}
		throw Error("unclosed [");
	}

	// For unusual identifiers, standard SQL uses double quotes
	void DoubleQuote() {
		Debug.Assert(text[textIndex] == '"');
		var i = textIndex + 1;
		var sb = new StringBuilder();
		while (i < text.Length) {
			switch (text[i]) {
			case '\\':
				switch (text[i + 1]) {
				case '"':
				case '\\':
					i++;
					break;
				}
				break;
			case '"':
				i++;
				switch (text[i]) {
				case '"':
					break;
				default:
					tokens.Add(new Token(textIndex, i, kQuotedName, sb.ToString().ToLowerInvariant()));
					textIndex = i;
					return;
				}
				break;
			}
			sb.Append(text[i++]);
		}
		throw Error("unclosed \"");
	}

	void LexWord() {
		Debug.Assert(IsWordPart(text[textIndex]));
		var i = textIndex;
		do
			i++;
		while (IsWordPart(text[i]));
		tokens.Add(new Token(textIndex, i, kWord, text[textIndex..i].ToLowerInvariant()));
		textIndex = i;
	}

	void LexNumber() {
		Debug.Assert(char.IsDigit(text[textIndex]) || text[textIndex] == '.');
		var i = textIndex;
		while (IsWordPart(text[i]))
			i++;
		if (text[i] == '.')
			do
				i++;
			while (IsWordPart(text[i]));
		tokens.Add(new Token(textIndex, i, kNumber, text[textIndex..i]));
		textIndex = i;
	}

	static bool IsWordPart(char c) {
		if (char.IsLetterOrDigit(c))
			return true;
		return c == '_';
	}

	// Error functions return exception objects instead of throwing immediately
	// so 'throw Error(...)' can mark the end of a case block
	Exception Error(string message, bool showToken = true) {
		if (tokenIndex < 0) {
			var location = new Location(file, text, textIndex);
			message = $"{location}: {message}";
		} else {
			var token = tokens[tokenIndex];
			var location = new Location(file, text, token.Start);
			if (showToken) {
				var s = text[token.Start..token.End];
				if (token.Type == -1)
					s = "EOF";
				message = $"{location}: {s}: {message}";
			} else
				message = $"{location}: {message}";
		}
		return new SqlError(message);
	}
}

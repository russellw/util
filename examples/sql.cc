/*
Copyright 2023 Russell Wallace
This file is part of Olivine.

Olivine is free software: you can redistribute it and/or modify it under the
terms of the GNU Affero General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

Olivine is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along
with Olivine.  If not, see <http:www.gnu.org/licenses/>.
*/

#include "olivine.h"

namespace olivine {
namespace {
enum {
	k_comment = ntoks,
	k_quote,
};

struct Parser1: Parser {
	SqlSchema& schema;

	// tokenizer
	void lex() {
		for (;;) {
			auto s = tokBegin = src;
			switch (*s) {
			case ' ':
			case '\f':
			case '\n':
			case '\r':
			case '\t':
				src = s + 1;
				continue;
			case '-':
				if (s[1] == '-') {
					s = strchr(s, '\n');
					str.assign(src, s);
					src = s;
					tok = k_comment;
					return;
				}
				break;
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
			case 'N':
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
				do {
					*s = tolower1(*s);
					++s;
				} while (isid(*s));
				str.assign(src, s);
				src = s;
				tok = k_word;
				return;
			case '\'':
				++s;
				for (;;) {
					if (*s == '\'') {
						if (s[1] != '\'')
							break;
						++s;
					}
					if (*s == '\n')
						err("unclosed quote");
					++s;
				}
				++s;
				str.assign(src, s);
				src = s;
				tok = k_quote;
				return;
			case '\\':
				s = strchr(s, '\n');
				str.assign(src, s);
				src = s;
				tok = k_comment;
				return;
			case 0:
				tok = 0;
				return;
			}
			src = s + 1;
			tok = *s;
			return;
		}
	}

	// parser
	bool eat(int k) {
		if (tok == k) {
			lex();
			return 1;
		}
		return 0;
	}

	bool eat(const char* s) {
		if (tok == k_word && str == s) {
			lex();
			return 1;
		}
		return 0;
	}

	void expect(char k) {
		if (!eat(k)) {
			sprintf(buf, "expected '%c'", k);
			err(buf);
		}
	}

	void expect(const char* s) {
		if (!eat(s)) {
			sprintf(buf, "expected '%s'", s);
			err(buf);
		}
	}

	void expectWord() {
		if (tok != k_word)
			err("expected word");
		lex();
	}

	void semi() {
		auto s = src;
		while (*s != ';') {
			if (!*s)
				err("missing ';'");
			++s;
		}
		src = s;
		lex();
	}

	Parser1(const char* file, SqlSchema& schema): Parser(file), schema(schema) {
		lex();
		for (;;)
			switch (tok) {
			case 0:
				return;
			case k_comment:
				schema.header.push_back(str);
				lex();
				break;
			case k_word:
				if (str == "create") {
					lex();
					if (str == "table") {
						lex();
						auto table = new SqlTable;

						table->name = str;
						expectWord();

						expect('(');
						do {
							auto column = new SqlColumn;

							column->name = str;
							expectWord();

							// type
							column->type = str;
							expectWord();
							if (eat('(')) {
								column->size = str;
								expectWord();
								expect(')');
							}
							if (eat("generated")) {
								expect("always");
								expect("as");
								expect("identity");
								column->generated = 1;
							}

							// primary key
							if (eat("primary")) {
								expect("key");
								column->primaryKey = 1;
							}

							// foreign key
							if (eat("references")) {
								column->referencesTableName = str;
								expectWord();
								expect('(');
								column->referencesColumnName = str;
								expectWord();
								expect(')');
							}

							table->columns.push_back(column);
						} while (eat(','));
						expect(')');
						expect(';');

						schema.tables.push_back(table);
						break;
					}
					if (str == "database") {
						auto s = src;
						semi();
						schema.header.push_back(string("CREATE DATABASE") + string(s, src));
						lex();
						break;
					}
					err("unknown noun");
				}
				if (str == "drop") {
					lex();
					if (str == "database") {
						auto s = src;
						semi();
						schema.header.push_back(string("DROP DATABASE") + string(s, src));
						lex();
						break;
					}
					err("unknown noun");
				}
				err("unknown verb");
			default:
				err("unknown syntax");
			}
	}
};

template <class T> void topologicalSortRecur(const vector<T>& v, vector<T>& r, unordered_set<T>& visited, T a) {
	if (visited.count(a))
		return;
	visited.insert(a);
	for (auto b: a->links)
		topologicalSortRecur(v, r, visited, b);
	r.push_back(a);
}

template <class T> void topologicalSort(vector<T>& v) {
	unordered_set<T> visited;
	vector<T> r;
	for (auto a: v)
		topologicalSortRecur(v, r, visited, a);
	v = r;
}
} // namespace

void readSql(const char* file, SqlSchema& schema) {
	Parser1 parser(file, schema);

	// link table references
	unordered_map<string, SqlTable*> tables;
	for (auto table: schema.tables)
		tables[table->name] = table;
	for (auto table: schema.tables)
		for (auto column: table->columns)
			if (column->referencesTableName.size()) {
				column->referencesTable = tables.at(column->referencesTableName);
				table->links.push_back(column->referencesTable);
			}

	// make it valid SQL with no forward references
	topologicalSort(schema.tables);
}
} // namespace olivine

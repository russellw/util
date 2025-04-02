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

#include <regex>
using std::regex;
using std::smatch;

#include <olivine.h>
using namespace olivine;

#ifndef NDEBUG
#ifdef _WIN32
#include <windows.h>

LONG WINAPI handler(_EXCEPTION_POINTERS* ExceptionInfo) {
	if (ExceptionInfo->ExceptionRecord->ExceptionCode == EXCEPTION_STACK_OVERFLOW)
		WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), "stack overflow\n", 15, 0, 0);
	else {
		printf("Exception code %lx\n", ExceptionInfo->ExceptionRecord->ExceptionCode);
		stackTrace();
		fflush(stdout);
	}
	ExitProcess(ExceptionInfo->ExceptionRecord->ExceptionCode);
}
#endif
#endif

// SORT
regex assignRegex(R"((\w+) = )");
regex commentRegex(R"(\s*//.*)");
regex emptyCommentRegex(R"(\s*//)");
regex fnBraceRegex(R"((\w+)\(.*\{$)");
regex fnRegex(R"((\w+)\()");
regex rbraceNamespaceRegex(R"(\} // namespace.*)");
regex rbraceRegex(R"(\s*\})");
regex rbraceSemiRegex(R"(\s*\};)");
regex sortCommentRegex(R"(\s*// SORT)");
regex structBraceRegex(R"((\w+) \{$)");
regex varRegex(R"((\w+)[;,])");
//

vector<string> v;

string rbrace = "}";
const string& at(size_t i) {
	if (i < v.size())
		return v[i];
	return rbrace;
}

struct Block {
	size_t first, last;
	string key;

	Block(const char* file, int dent, size_t i): first(i) {
		while (regex_match(at(i), commentRegex))
			++i;
		auto& s = at(i);
		smatch m;
		if (regex_search(s, m, fnBraceRegex)) {
			key = m[1].str() + ':' + s;
			do {
				++i;
				if (i == v.size()) {
					printf("%s:%zu: unclosed function\n", file, first + 1);
					exit(1);
				}
			} while (!(indent(v, i) == dent && regex_match(at(i), rbraceRegex)));
			last = i + 1;
			return;
		}
		if (regex_search(s, m, structBraceRegex)) {
			key = m[1].str() + ':' + s;
			do {
				++i;
				if (i == v.size()) {
					printf("%s:%zu: unclosed definition\n", file, first + 1);
					exit(1);
				}
			} while (!(indent(v, i) == dent && regex_match(at(i), rbraceSemiRegex)));
			last = i + 1;
			return;
		}
		if (regex_search(s, m, assignRegex) || regex_search(s, m, fnRegex) || regex_search(s, m, varRegex)) {
			key = m[1].str() + ':' + s;
			last = i + 1;
			return;
		}
		printf("%s:%zu: unknown syntax\n", file, i + 1);
		exit(1);
	}

	bool operator<(const Block& b) {
		return key < b.key;
	}

	void to(vector<string>& r) {
		r.insert(r.end(), v.begin() + first, v.begin() + last);
	}
};

int main(int argc, char** argv) {
#ifndef NDEBUG
#ifdef _WIN32
	AddVectoredExceptionHandler(0, handler);
#endif
#endif
	try {
		for (int i = 1; i < argc; ++i) {
			auto file = argv[i];
			if (*file == '-') {
				puts("sort-c file...");
				return 0;
			}
			readLines(file, v);
			auto old = v;

			for (size_t i = 0; i < v.size();) {
				if (!regex_match(v[i], sortCommentRegex)) {
					++i;
					continue;
				}

				// sortable blocks should be indented at the same level as the marker comment
				auto dent = indent(v, i);
				++i;

				// get group of blocks
				size_t j = i;
				vector<Block> blocks;
				for (;;) {
					// skip intervening blank lines
					while (at(j).empty())
						++j;

					// end of group?
					if (indent(v, j) < dent)
						break;
					auto& s = v[j];
					if (regex_match(s, emptyCommentRegex))
						break;
					if (regex_match(s, sortCommentRegex))
						break;
					if (regex_match(s, rbraceNamespaceRegex))
						break;

					// get the next block
					Block block(file, dent, j);
					j = block.last;
					blocks.push_back(block);
				}

				// sort
				sort(blocks.begin(), blocks.end());

				// if blocks are multiline, separate with blank lines
				bool blanks = 0;
				for (auto block: blocks)
					if (block.last - block.first > 1) {
						blanks = 1;
						break;
					}

				// update
				vector<string> r;
				for (auto block: blocks) {
					if (blanks && r.size())
						r.push_back("");
					block.to(r);
				}
				if (blanks && regex_match(at(j), commentRegex))
					r.push_back("");
				v.erase(v.begin() + i, v.begin() + j);
				v.insert(v.begin() + i, r.begin(), r.end());

				i += r.size();
			}

			if (old != v)
				writeLines(file, v);
		}
		return 0;
	} catch (exception& e) {
		println(e.what());
		return 1;
	}
}

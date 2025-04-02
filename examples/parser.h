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

// this, and its effective sub-enums in specific parsers, are not enum classes because enumerated tokens will be freely mixed with
// literal characters
enum {
	k_word = 0x100,
	ntoks
};

// defining our own isxxxxx in preference to the ones in ctype, avoids surprising behavior related to locale, and undefined behavior
// if an input has the high bit set, and you forget to cast to unsigned char
inline bool isdigit1(int c) {
	return '0' <= c && c <= '9';
}

inline bool islower1(int c) {
	return 'a' <= c && c <= 'z';
}

inline bool isupper1(int c) {
	return 'A' <= c && c <= 'Z';
}

inline bool isalpha1(int c) {
	return islower1(c) || isupper1(c);
}

inline bool isalnum1(int c) {
	return isalpha1(c) || isdigit1(c);
}

inline bool isid(int c) {
	return isalnum1(c) || c == '_';
}

inline int tolower1(int c) {
	return isupper1(c) ? c + 32 : c;
}

inline int toupper1(int c) {
	return islower1(c) ? c - 32 : c;
}

struct Parser {
	// full name of the file being parsed
	const char* file;

	// source text
	string text;

	// beginning of current token in source text
	char* tokBegin;

	// current position in source text
	char* src;

	// current token, as direct char for single-char tokens, or language-specific enum otherwise
	int tok;

	// current token keyword or identifier
	string str;

	Parser(const char* file);

	// report an error with line number, and exit
	[[noreturn]] void err(const char* msg);
};

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
Parser::Parser(const char* file): file(file), text(readFile(file)) {
	src = text.data();
}

void Parser::err(const char* msg) {
	size_t line = 1;
	for (auto s = text.data(); s < tokBegin; ++s)
		if (*s == '\n')
			++line;
	printf("%s:%zu: %s\n", file, line, msg);
	exit(1);
}
} // namespace olivine

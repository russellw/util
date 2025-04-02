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

#include <olivine.h>
using namespace olivine;

int main(int argc, char** argv) {
	for (int i = 1; i < argc; ++i) {
		auto file = argv[i];
		if (*file == '-') {
			puts("fmt-schema file...");
			return 0;
		}
		SqlSchema schema;
		readSql(file, schema);

		string o;
		for (auto& s: schema.header) {
			o += s;
			o += '\n';
		}
		for (auto table: schema.tables) {
			o += "CREATE TABLE ";
			o += table->name;
			o += "(\n";
			for (auto column: table->columns) {
				o += '\t';
				o += column->name;
				o += ' ';

				// type
				for (auto c: column->type)
					o += toupper1(c);
				if (column->size.size()) {
					o += '(';
					o += column->size;
					o += ')';
				}
				if (column->generated)
					o += " GENERATED ALWAYS AS IDENTITY";

				// primary key
				if (column->primaryKey)
					o += " PRIMARY KEY";

				// foreign key
				if (column->referencesTableName.size()) {
					o += " REFERENCES ";
					o += column->referencesTableName;
					o += '(';
					o += column->referencesColumnName;
					o += ')';
				}

				if (column != table->columns.back())
					o += ',';
				o += '\n';
			}
			o += ");\n";
		}

		if (readFile(file) != o)
			writeFile(file, o);
	}
	return 0;
}

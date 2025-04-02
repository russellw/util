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
	if (argc < 2) {
		puts("compile-schema file.sql\n"
			 "writes schema.hxx, schema.cxx");
		return 1;
	}

	SqlSchema schema;
	readSql(argv[1], schema);

	// header
	string o = "// AUTO GENERATED - DO NOT EDIT\n";

	for (auto table: schema.tables) {
		o += "extern Table ";
		o += table->name;
		o += "_table;\n";
	}

	o += "extern Table* tables[];\n";

	writeFile("schema.hxx", o);

	// definitions
	o = "// AUTO GENERATED - DO NOT EDIT\n";
	o += "#include <olivine.h>\n";
	o += "using namespace olivine;\n";
	o += "#include \"schema.hxx\"\n";

	for (auto table: schema.tables) {
		o += "Column ";
		o += table->name;
		o += "_columns[] = {\n";
		for (auto column: table->columns) {
			o += '{';
			o += quote(column->name);

			// type
			o += ',';
			o += "Type::" + column->type;
			if (column->type == "char")
				o += 's';
			o += ',';
			if (column->size.size())
				o += column->size;
			else
				o += '0';
			o += ',';
			o += '0' + column->generated;

			// primary key
			o += ',';
			o += '0' + column->primaryKey;

			// foreign key
			if (column->referencesTable) {
				o += ',';
				o += '&';
				o += column->referencesTable->name;
				o += "_table";
			}

			o += "},\n";
		}
		o += "0\n";
		o += "};\n";

		o += "Table ";
		o += table->name;
		o += "_table = {\"";
		o += table->name;
		o += "\",";
		o += table->name;
		o += "_columns};\n";
	}

	o += "Table* tables[] = {\n";
	for (auto table: schema.tables) {
		o += '&';
		o += table->name;
		o += "_table,\n";
	}
	o += "0\n";
	o += "};\n";

	writeFile("schema.cxx", o);
	return 0;
}

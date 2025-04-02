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

#ifndef NDEBUG

#ifdef _WIN32
#include <windows.h>

#include <crtdbg.h>
#include <dbghelp.h>
#else
extern "C" const char* __asan_default_options() {
	return "detect_leaks=0";
}
#endif

namespace olivine {
void stackTrace() {
#ifdef _WIN32
	// process
	auto process = GetCurrentProcess();
	SymInitialize(process, 0, 1);

	// stack frames
	const int maxFrames = 100;
	void* stack[maxFrames];
	auto nframes = CaptureStackBackTrace(1, maxFrames, stack, 0);

	// symbol
	static_assert(sizeof(SYMBOL_INFO) + MAX_SYM_NAME < sizeof buf);
	auto si = (SYMBOL_INFO*)buf;
	si->SizeOfStruct = sizeof(SYMBOL_INFO);
	si->MaxNameLen = MAX_SYM_NAME;

	// location
	IMAGEHLP_LINE64 loc;
	loc.SizeOfStruct = sizeof loc;

	// print
	for (int i = 0; i < nframes; ++i) {
		auto addr = (DWORD64)(stack[i]);
		SymFromAddr(process, addr, 0, si);
		DWORD displacement;
		if (SymGetLineFromAddr64(process, addr, &displacement, &loc))
			printf("%s:%lu: ", loc.FileName, loc.LineNumber);
		printf("%s\n", si->Name);
	}
#endif
}
} // namespace olivine

#endif

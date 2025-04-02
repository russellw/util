#include <olivine.h>

#include "main.h"

#include <new>

#ifdef _WIN32
#include <windows.h>

namespace {
LONG WINAPI handler(struct _EXCEPTION_POINTERS* ExceptionInfo) {
	if (ExceptionInfo->ExceptionRecord->ExceptionCode == EXCEPTION_STACK_OVERFLOW)
		WriteFile(GetStdHandle(STD_ERROR_HANDLE), "Stack overflow\n", 15, 0, 0);
	else {
		fprintf(stderr, "Exception code %lx\n", ExceptionInfo->ExceptionRecord->ExceptionCode);
		stackTrace();
	}
	ExitProcess(1);
}
} // namespace
#else
#include <unistd.h>
#endif

int main(int argc, char** argv) {
	//init
	std::set_new_handler([]() {
		perror("new");
		exit(1);
	});
#ifdef _WIN32
	AddVectoredExceptionHandler(0, handler);
#endif

	//command line
	bool dump = 0;
	vector<char*> files;
	for (int i = 1; i != argc; ++i) {
		auto s = argv[i];
		if (*s == '-') {
			while (*s == '-') ++s;
			switch (*s) {
			case 'h':
				puts("-h  Show help\n"
					 "-V  Show version"
					 "-x  Dump AST");
				return 0;
			case 'x':
				dump = 1;
				continue;
			}
			fprintf(stderr, "%s: unknown option\n", argv[i]);
			return 1;
		}
		files.push_back(s);
	}

	//dump AST
	if (dump) {
		for (auto file: files) {
			vector<dyn> v;
			parse(file, v);
			puts("[");
			for (dyn a: v) {
				print(a);
				putchar('\n');
			}
			puts("]");
		}
	}
	return 0;
}

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
using namespace std;

#ifdef NDEBUG
#define debug(a)
#else
#define debug(a) cout << __FILE__ << ':' << __LINE__ << ": " << __func__ << ": " << #a << ": " << (a) << '\n'
#endif

void test(string program) {
	auto start = chrono::high_resolution_clock::now();
	auto cmd = program + " < data.tsv";
#ifndef _WIN32
	cmd = "./" + cmd;
#endif
	if (system(cmd.data()))
		throw runtime_error(program + " failed");
	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
	cout << program << '\t' << duration.count() << '\n';
}

int main(int argc, char** argv) {
	try {
		test("test-get");
		test("test-getchar");
		test("test-getline");
		test("test-istream_iterator");
		test("test-istreambuf_iterator");
		return 0;
	} catch (exception& e) {
		cerr << e.what() << '\n';
		return 1;
	}
}

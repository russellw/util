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

int main(int argc, char** argv) {
	try {
		string s;
		char c;
		while (cin.get(c))
			s += c;
		if (s.size() != 157777780)
			throw runtime_error(to_string(s.size()));
		return 0;
	} catch (exception& e) {
		cerr << e.what() << '\n';
		return 1;
	}
}

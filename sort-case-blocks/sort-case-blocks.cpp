#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
using std::cerr;
using std::cout;
using std::hash;
using std::ostream;
using std::pair;
using std::runtime_error;
using std::string;
using std::unordered_map;
using std::unordered_set;
using std::vector;

// Helper function to create detailed error messages
namespace message_detail {
template <typename T> string toString(const T& val) {
	std::ostringstream oss;
	oss << val;
	return oss.str();
}

inline string makeAssertMessage(const char* expression, const char* file, int line, const string& message = "") {
	std::ostringstream oss;
	oss << "Assertion failed: " << expression << "\nFile: " << file << "\nLine: " << line;

	if (!message.empty()) {
		oss << "\nMessage: " << message;
	}

	return oss.str();
}
} // namespace message_detail

// Basic assertion that throws std::runtime_error
#define ASSERT(condition)                                                                                                          \
	do {                                                                                                                           \
		if (!(condition)) {                                                                                                        \
			throw runtime_error(message_detail::makeAssertMessage(#condition, __FILE__, __LINE__));                                \
		}                                                                                                                          \
	} while (0)

#define dbg(a) std::cout << __FILE__ << ':' << __LINE__ << ": " << (a) << '\n'

#ifdef _WIN32
#include <windows.h>

LONG WINAPI unhandledExceptionFilter(EXCEPTION_POINTERS* exInfo) {
	cerr << "Unhandled exception: " << std::hex << exInfo->ExceptionRecord->ExceptionCode << '\n';
	return EXCEPTION_EXECUTE_HANDLER;
}
#endif

// Count leading tabs in a string
int countLeadingTabs(const std::string& str) {
	int count = 0;
	for (char c : str) {
		if (c == '\t') {
			count++;
		} else {
			break;
		}
	}
	return count;
}

bool endsWith(const string& s, int c) {
	return s.size() && s.back() == c;
}

std::vector<std::string> lines;

int dent;

int indent(int i) {
	ASSERT(i < lines.size());
	if (lines[i].empty()) {
		return 1000000000;
	}
	return countLeadingTabs(lines[i]);
}

bool isSwitch(int i) {
	auto line = lines[i];
	size_t pos = line.find_first_not_of('\t');
	if (pos == std::string::npos) {
		return false;
	}

	if (line.substr(pos, 7) == "switch ") {
		dent = pos;
		return true;
	}

	return false;
}

bool isCase(int i) {
	ASSERT(dent);
	ASSERT(i < lines.size());
	auto line = lines[i];
	size_t pos = line.find_first_not_of('\t');
	if (pos == std::string::npos) {
		return false;
	}

	return line.substr(pos, 5) == "case " || line.substr(pos, 8) == "default:";
}

int parseCase(int i) {
	ASSERT(isCase(i));
	bool brace = 0;
	while (isCase(i)) {
		if (endsWith(lines[i], '{')) {
			brace = 1;
		}
		i++;
	}
	while (indent(i) > dent) {
		i++;
	}
	ASSERT(indent(i) == dent);
	if (brace) {
		ASSERT(lines[i].substr(dent) == "}");
		i++;
	}
	return i;
}

struct Case {
	vector<string> v;

	Case(vector<string>::iterator first, vector<string>::iterator last): v(first, last) {
	}
};

void sortSwitch(int i) {
	ASSERT(isSwitch(i));
	i++;
	auto i0 = i;
	vector<Case*> cases;
	while (isCase(i)) {
		int j = parseCase(i);
		ASSERT(i < j);
		cases.push_back(new Case(lines.begin() + i, lines.begin() + j));
		i = j;
	}
	ASSERT(indent(i) == dent);
	ASSERT(lines[i].substr(dent) == "}");

	std::sort(cases.begin(), cases.end(), [](const Case* a, const Case* b) { return a->v[0] < b->v[0]; });

	i = i0;
	for (auto c : cases) {
		std::copy(c->v.begin(), c->v.end(), lines.begin() + i);
		i += c->v.size();
	}
	ASSERT(indent(i) == dent);
	ASSERT(lines[i].substr(dent) == "}");
}

void sortLines() {
	for (int i = 0; i < lines.size(); i++) {
		if (isSwitch(i)) {
			sortSwitch(i);
		}
	}
}

// Sort cases in switch statements within a file
std::string sortFileText(const std::string& content) {
	std::istringstream iss(content);
	std::string line;

	// Split content into lines
	while (std::getline(iss, line)) {
		// Remove Windows-style line endings if present
		if (!line.empty() && line.back() == '\r') {
			line.pop_back();
		}
		lines.push_back(line);
	}

	std::vector<std::string> result;
	size_t i = 0;

	while (i < lines.size()) {
	}

	// Join lines with UNIX line endings
	std::ostringstream oss;
	for (size_t i = 0; i < result.size(); ++i) {
		oss << result[i];
		oss << '\n';
	}

	return oss.str();
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
	SetUnhandledExceptionFilter(unhandledExceptionFilter);
#endif
	bool writeToFile = false;
	std::vector<std::string> filenames;

	// Parse command line arguments
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "-w") {
			writeToFile = true;
		} else {
			filenames.push_back(arg);
		}
	}

	if (filenames.empty()) {
		std::cerr << "Usage: " << argv[0] << " [-w] file1 [file2 ...]\n";
		return 1;
	}

	for (const auto& filename : filenames) {
		try {
			// Read the input file
			std::ifstream inFile(filename, std::ios::binary);
			if (!inFile) {
				std::cerr << "Error: Cannot open file " << filename << "\n";
				continue;
			}

			std::stringstream buffer;
			buffer << inFile.rdbuf();
			std::string content = buffer.str();
			inFile.close();

			// Sort the cases
			std::string sortedContent = sortFileText(content);

			// Check if there were any changes
			bool hasChanges = (content != sortedContent);

			if (writeToFile && hasChanges) {
				// Write to the input file
				std::ofstream outFile(filename, std::ios::binary | std::ios::trunc);
				if (!outFile) {
					std::cerr << "Error: Cannot write to file " << filename << "\n";
					continue;
				}
				outFile << sortedContent;
				outFile.close();
				std::cout << "Updated file: " << filename << "\n";
			} else if (!writeToFile) {
				// Print to stdout
				std::cout << sortedContent;
			} else {
				std::cout << "No changes needed for file: " << filename << "\n";
			}
		} catch (const std::exception& e) {
			std::cerr << "Error processing file " << filename << ": " << e.what() << "\n";
		}
	}

	return 0;
}

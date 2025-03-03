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

std::vector<std::string> lines;

bool isCase(int i) {
	ASSERT(i < lines.size());
	auto line = lines[i];
	size_t pos = line.find_first_not_of('\t');
	if (pos == std::string::npos) {
		return false;
	}

	return line.substr(pos, 5) == "case " || line.substr(pos, 8) == "default:";
}

std::string processContent(const std::string& content) {
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
	
	vector<string>result;
	for(int i=0;i<lines.size();i++)
	{
		if(i+1<lines.size()&&lines[i].empty()&&isCase(i+1))continue;
		result.push_back(lines[i]);
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
			std::string sortedContent = processContent(content);

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

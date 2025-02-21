#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

// Returns true if the line contains a case statement
bool isCase(const std::string& line) {
    size_t pos = line.find("case");
    if (pos == std::string::npos) return false;
    
    // Make sure "case" is at the start of the actual code (ignoring whitespace)
    while (pos > 0) {
        if (!std::isspace(line[pos-1])) return false;
        pos--;
    }
    return true;
}

// Returns true if the line contains an opening brace
bool hasOpeningBrace(const std::string& line) {
    return line.find("{") != std::string::npos;
}

// Removes opening brace and trailing whitespace from a line
std::string removeBrace(const std::string& line) {
    std::string result = line;
    size_t bracePos = result.find("{");
    if (bracePos != std::string::npos) {
        result = result.substr(0, bracePos);
    }
    // Remove trailing whitespace
    while (!result.empty() && std::isspace(result.back())) {
        result.pop_back();
    }
    return result;
}

// Gets the indentation level of a line
std::string getIndentation(const std::string& line) {
    size_t pos = 0;
    while (pos < line.length() && std::isspace(line[pos])) {
        pos++;
    }
    return line.substr(0, pos);
}

// Process a single file
bool processFile(const std::string& filename, bool writeToFile) {
    std::ifstream inFile(filename);
    if (!inFile) {
        std::cerr << "Error: Could not open input file: " << filename << std::endl;
        return false;
    }

    std::vector<std::string> lines;
    std::string line;
    
    // Read all lines
    while (std::getline(inFile, line)) {
        // Remove Windows line endings if present
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(line);
    }
    inFile.close();

    // Process the lines
    std::vector<std::string> output;
    bool modified = false;
    bool inCaseBlock = false;
    bool hasBrace = false;
    size_t lastCaseIndex = 0;
    std::string braceIndentation;

    for (size_t i = 0; i < lines.size(); i++) {
        if (isCase(lines[i])) {
            if (!inCaseBlock) {
                inCaseBlock = true;
                hasBrace = false;
            }
            if (hasOpeningBrace(lines[i])) {
                hasBrace = true;
                modified = true;
                braceIndentation = getIndentation(lines[i]);
                output.push_back(removeBrace(lines[i]));
            } else {
                output.push_back(lines[i]);
            }
            lastCaseIndex = output.size() - 1;
        } else {
            if (inCaseBlock) {
                // Check if this line starts a new block or ends the case block
                if (!std::all_of(lines[i].begin(), lines[i].end(), ::isspace)) {
                    if (hasBrace) {
                        // Insert the brace after the last case statement
                        output.insert(output.begin() + lastCaseIndex + 1, braceIndentation + "{");
                    }
                    inCaseBlock = false;
                }
            }
            output.push_back(lines[i]);
        }
    }

    // Handle case block at end of file
    if (inCaseBlock && hasBrace) {
        output.insert(output.begin() + lastCaseIndex + 1, braceIndentation + "{");
    }

    if (!modified) {
        return true;
    }

    if (writeToFile) {
        std::ofstream outFile(filename, std::ios::binary);
        if (!outFile) {
            std::cerr << "Error: Could not open output file: " << filename << std::endl;
            return false;
        }
        for (size_t i = 0; i < output.size(); i++) {
            outFile << output[i] << '\n';  // Use Unix line endings
        }
        std::cout << "Updated " << filename << std::endl;
    } else {
        for (const auto& line : output) {
            std::cout << line << '\n';
        }
    }

    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [-w] file1 [file2 ...]" << std::endl;
        return 1;
    }

    bool writeToFile = false;
    int fileStartIndex = 1;

    if (std::string(argv[1]) == "-w") {
        writeToFile = true;
        fileStartIndex = 2;
        if (argc < 3) {
            std::cerr << "Error: No input files specified" << std::endl;
            return 1;
        }
    }

    bool success = true;
    for (int i = fileStartIndex; i < argc; i++) {
        if (!processFile(argv[i], writeToFile)) {
            success = false;
        }
    }

    return success ? 0 : 1;
}
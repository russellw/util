#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>

using namespace std;

// Function to read a file's content into a string
string readFile(const string &filename) {
    ifstream file(filename, ios::binary);
    if (!file) {
        throw runtime_error("Could not open file: " + filename);
    }
    stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Function to write content to a file
void writeFile(const string &filename, const string &content) {
    ofstream file(filename, ios::binary);
    if (!file) {
        throw runtime_error("Could not write to file: " + filename);
    }
    file << content;
}

// Function to split a string into lines
vector<string> splitLines(const string &content) {
    vector<string> lines;
    stringstream ss(content);
    string line;
    while (getline(ss, line)) {
        lines.push_back(line);
    }
    return lines;
}

// Function to join lines into a single string
string joinLines(const vector<string> &lines) {
    string content;
    for (const auto &line : lines) {
        content += line + "\n";
    }
    return content;
}

// Function to sort case blocks alphabetically
vector<string> sortCases(const vector<string> &lines, size_t &index) {
    vector<pair<string, vector<string>>> cases;
    string currentCase;
    vector<string> currentBlock;
    size_t baseIndent = lines[index].find_first_not_of('\t');

    while (index < lines.size()) {
        string line = lines[index];
        size_t indent = line.find_first_not_of('\t');

        if (indent == baseIndent && line.find("case ") == 0) {
            if (!currentBlock.empty()) {
                cases.emplace_back(currentCase, currentBlock);
            }
            currentCase = line;
            currentBlock.clear();
        } else if (indent == baseIndent && line == "}") {
            if (!currentBlock.empty()) {
                cases.emplace_back(currentCase, currentBlock);
            }
            break;
        } else {
            currentBlock.push_back(line);
        }
        index++;
    }

    // Sort cases alphabetically by the first line of each case
    sort(cases.begin(), cases.end(), [](const auto &a, const auto &b) {
        return a.first < b.first;
    });

    // Flatten sorted cases back into a single vector of lines
    vector<string> sortedLines;
    for (const auto &caseBlock : cases) {
        sortedLines.push_back(caseBlock.first);
        sortedLines.insert(sortedLines.end(), caseBlock.second.begin(), caseBlock.second.end());
    }
    sortedLines.push_back("}");
    return sortedLines;
}

// Function to process the content of a file and sort cases
string processContent(const string &content) {
    vector<string> lines = splitLines(content);
    vector<string> result;

    for (size_t i = 0; i < lines.size(); i++) {
        if (lines[i].find("switch (") != string::npos) {
            result.push_back(lines[i]);
            i++;
            auto sortedBlock = sortCases(lines, i);
            result.insert(result.end(), sortedBlock.begin(), sortedBlock.end());
        } else {
            result.push_back(lines[i]);
        }
    }
    return joinLines(result);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " [-w] <file1> [file2 ...]" << endl;
        return 1;
    }

    bool overwrite = false;
    vector<string> files;

    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "-w") {
            overwrite = true;
        } else {
            files.push_back(argv[i]);
        }
    }

    for (const auto &file : files) {
        try {
            string originalContent = readFile(file);
            string processedContent = processContent(originalContent);

            if (overwrite) {
                if (originalContent != processedContent) {
                    writeFile(file, processedContent);
                    cout << "File " << file << " updated." << endl;
                } else {
                    cout << "File " << file << " unchanged." << endl;
                }
            } else {
                cout << processedContent;
            }
        } catch (const exception &e) {
            cerr << "Error processing file " << file << ": " << e.what() << endl;
        }
    }

    return 0;
}

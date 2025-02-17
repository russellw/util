#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>

using namespace std;

// --- Utility functions ---

// Remove whitespace from both ends of a string.
string trim(const string &s) {
    size_t start = 0;
    while (start < s.size() && isspace((unsigned char)s[start]))
        start++;
    size_t end = s.size();
    while (end > start && isspace((unsigned char)s[end - 1]))
        end--;
    return s.substr(start, end - start);
}

// Count the number of leading tab characters.
int countLeadingTabs(const string &s) {
    int count = 0;
    for (char c : s) {
        if (c == '\t')
            count++;
        else
            break;
    }
    return count;
}

// Extract the function name from a signature line.
// We assume that the signature line (e.g. "runtime_error error(const string& msg) const {")
// contains a '(' and that the function name is the last token before it.
string extractFunctionName(const string &signature) {
    size_t pos = signature.find('(');
    if (pos == string::npos)
        return "";
    string beforeParen = signature.substr(0, pos);
    beforeParen = trim(beforeParen);
    size_t lastSpace = beforeParen.find_last_of(" \t");
    if (lastSpace == string::npos)
        return beforeParen;
    return beforeParen.substr(lastSpace + 1);
}

// --- Structure for a function definition block ---

struct FunctionBlock {
    vector<string> lines;  // the complete block (including any attached comments)
    string keyName;        // function name for sorting
    string keyFull;        // the complete signature line (tiebreaker)
};

// --- Main program ---

// The program scans each file for a marker line that (after trimming) is "// SORT FUNCTIONS".
// The sorted section is assumed to start immediately after that line and continue until the first
// nonblank line whose indent is less than that of the marker.
// Within that section, each function block is assumed to start at a line with the same indentation
// as the marker. Any immediately preceding comment lines (also at that indentation) are attached.
// A function block ends when a line is encountered that is (after trimming) exactly "}" and has the same indent.
// Functions are sorted by name (and, in case of overloads, by the complete signature line).
// Finally, the section is rebuilt so that each function block is preceded by exactly one blank line.
int main(int argc, char* argv[]) {
    bool writeMode = false;
    vector<string> filenames;
    
    // Process command-line options.
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-w") {
            writeMode = true;
        } else {
            filenames.push_back(arg);
        }
    }
    
    if (filenames.empty()) {
        cerr << "Usage: " << argv[0] << " [-w] file1 [file2 ...]" << endl;
        return 1;
    }
    
    // Process each file.
    for (auto &filename : filenames) {
        ifstream ifs(filename);
        if (!ifs) {
            cerr << "Error opening file: " << filename << endl;
            continue;
        }
        
        // Read the entire file into a vector of lines (normalize line endings to UNIX style).
        vector<string> lines;
        {
            string line;
            while(getline(ifs, line)) {
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                lines.push_back(line);
            }
        }
        ifs.close();
        
        // Find the marker line "// SORT FUNCTIONS" and its indentation.
        bool foundMarker = false;
        int markerIndex = -1;
        int baseIndent = 0;
        for (size_t i = 0; i < lines.size(); i++) {
            if (trim(lines[i]) == "// SORT FUNCTIONS") {
                foundMarker = true;
                markerIndex = i;
                baseIndent = countLeadingTabs(lines[i]);
                break;
            }
        }
        
        // If no marker is found, leave the file unchanged.
        if (!foundMarker) {
            if (!writeMode) {
                for (auto &l : lines)
                    cout << l << "\n";
            }
            continue;
        }
        
        // Identify the sorted section.
        // It starts at the line immediately after the marker and continues until a nonblank line
        // is encountered that is indented less than the marker.
        int sortedStart = markerIndex + 1;
        int sortedEnd = sortedStart;
        while (sortedEnd < (int)lines.size()) {
            string trimmedLine = trim(lines[sortedEnd]);
            if (!trimmedLine.empty() && countLeadingTabs(lines[sortedEnd]) < baseIndent)
                break;
            sortedEnd++;
        }
        
        // Copy out the sorted section.
        vector<string> sortedSection;
        for (int i = sortedStart; i < sortedEnd; i++) {
            sortedSection.push_back(lines[i]);
        }
        
        // --- Parse function definitions from the sorted section using indentation ---
        // We use a state machine to group lines:
        // - Lines with the same indent as the marker are candidates for the start of a function block.
        // - Any immediately preceding comment lines (starting with "//") at that indent are attached.
        // - A function block ends when we see a line with baseIndent whose trimmed content is "}".
        vector<FunctionBlock> blocks;
        vector<string> pending;      // temporarily holds attached comments.
        vector<string> currentBlock; // holds lines of the current function block.
        bool inBlock = false;
        
        for (size_t idx = 0; idx < sortedSection.size(); idx++) {
            string line = sortedSection[idx];
            string trimmedLine = trim(line);
            
            // Handle blank lines.
            if (trimmedLine.empty()) {
                if (!inBlock)
                    pending.clear(); // blank line resets any pending attached comments.
                else
                    currentBlock.push_back(line);
                continue;
            }
            
            int indent = countLeadingTabs(line);
            
            if (!inBlock) {
                // Not currently in a function block.
                if (indent == baseIndent) {
                    // At base indent.
                    if (trimmedLine.rfind("//", 0) == 0) {
                        // It's a comment line; add to pending.
                        pending.push_back(line);
                    } else {
                        // This is the function signature.
                        currentBlock = pending;  // attach any preceding comments.
                        pending.clear();
                        currentBlock.push_back(line);
                        inBlock = true;
                    }
                } else {
                    // Line not at base indent and not in block: skip it.
                }
            } else {
                // We are inside a function block; add the line.
                currentBlock.push_back(line);
                // If this line is a closing brace at base indent, we finish the block.
                if (indent == baseIndent && trimmedLine == "}") {
                    // Identify the function signature line: the first non-comment line.
                    string keyFull, keyName;
                    for (auto &blkLine : currentBlock) {
                        string noTabs = blkLine;
                        while (!noTabs.empty() && noTabs[0] == '\t')
                            noTabs.erase(0, 1);
                        if (noTabs.size() >= 2 && noTabs.substr(0, 2) == "//")
                            continue;
                        keyFull = blkLine;
                        keyName = extractFunctionName(blkLine);
                        break;
                    }
                    blocks.push_back({ currentBlock, keyName, keyFull });
                    currentBlock.clear();
                    inBlock = false;
                }
            }
        }
        
        // If no function blocks were found, leave the section unchanged.
        if (blocks.empty()) {
            if (!writeMode) {
                for (auto &l : lines)
                    cout << l << "\n";
            }
            continue;
        }
        
        // --- Sort the function blocks ---
        sort(blocks.begin(), blocks.end(), [](const FunctionBlock &a, const FunctionBlock &b) {
            if (a.keyName == b.keyName)
                return a.keyFull < b.keyFull;
            return a.keyName < b.keyName;
        });
        
        // --- Rebuild the sorted section ---
        // Each function block is output with exactly one blank line preceding it.
        vector<string> newSortedSection;
        for (auto &fb : blocks) {
            newSortedSection.push_back("");  // exactly one blank line
            for (auto &l : fb.lines)
                newSortedSection.push_back(l);
        }
        
        // --- Reassemble the file ---
        // The new file consists of:
        // 1. Everything up to (and including) the marker.
        // 2. The new sorted section.
        // 3. The remainder of the file.
        vector<string> finalLines;
        for (int i = 0; i < sortedStart; i++)
            finalLines.push_back(lines[i]);
        for (auto &l : newSortedSection)
            finalLines.push_back(l);
        for (int i = sortedEnd; i < (int)lines.size(); i++)
            finalLines.push_back(lines[i]);
        
        // Build the final content using UNIX newlines.
        ostringstream ossNew;
        for (auto &l : finalLines)
            ossNew << l << "\n";
        string newContent = ossNew.str();
        
        // Also reassemble the original file (normalized) for comparison.
        ostringstream ossOrig;
        for (auto &l : lines)
            ossOrig << l << "\n";
        string origContent = ossOrig.str();
        
        // If there is no change and -w was specified, do nothing.
        if (newContent == origContent) {
            if (!writeMode)
                cout << newContent;
        } else {
            if (writeMode) {
                ofstream ofs(filename);
                if (!ofs)
                    cerr << "Error writing to file: " << filename << "\n";
                else {
                    ofs << newContent;
                    ofs.close();
                }
            } else {
                cout << newContent;
            }
        }
    }
    
    return 0;
}

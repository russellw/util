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

// The program scans each file for a marker line (after trimming) equal to "// SORT FUNCTIONS".
// The sorted section starts immediately after the marker and continues until the first
// nonblank line whose indent is less than that of the marker.
// Within that section, a function block is defined as follows:
//  - Lines at base indent that start with "//" are considered attached comments.
//  - A function block begins with a non-comment line at base indent (the signature line),
//    along with any immediately preceding comment lines.
//  - The function block is assumed to end when a line at base indent is encountered whose
//    trimmed content is exactly "}".
// After sorting the function blocks (by function name, and by signature for ties), the sorted
// section is reassembled so that each block is preceded by exactly one blank line.
// Finally, if the line that ended the sorted section (if any) does not start (after removing tabs)
// with a '}', we add an extra blank line after the sorted section.
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
        // Use a simple state machine:
        //  - Pending comment lines at base indent are stored.
        //  - When a non-comment line at base indent is encountered, a function block begins (attaching pending comments).
        //  - The block continues until a line at base indent with trimmed content "}" is encountered.
        vector<FunctionBlock> blocks;
        vector<string> pending;      // holds attached comment lines
        vector<string> currentBlock; // holds lines of the current function block
        bool inBlock = false;
        
        for (size_t idx = 0; idx < sortedSection.size(); idx++) {
            string line = sortedSection[idx];
            string trimmedLine = trim(line);
            
            // Handle blank lines.
            if (trimmedLine.empty()) {
                if (!inBlock)
                    pending.clear(); // reset pending comments on blank line
                else
                    currentBlock.push_back(line);
                continue;
            }
            
            int indent = countLeadingTabs(line);
            
            if (!inBlock) {
                // Not currently in a function block.
                if (indent == baseIndent) {
                    if (trimmedLine.rfind("//", 0) == 0) {
                        // It's a comment line; add to pending.
                        pending.push_back(line);
                    } else {
                        // This is the function signature.
                        currentBlock = pending;  // attach any preceding comments
                        pending.clear();
                        currentBlock.push_back(line);
                        inBlock = true;
                    }
                } // else: ignore lines not at base indent.
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
        
        // --- Check the line that ended the sorted section ---
        // If sortedEnd is not the end of the file, dedent the line and check whether its first character is '}'
        bool needExtraBlank = false;
        if (sortedEnd < (int)lines.size()) {
            string dedented = lines[sortedEnd];
            while (!dedented.empty() && dedented[0] == '\t')
                dedented.erase(0, 1);
            if (dedented.empty() || dedented[0] != '}') {
                needExtraBlank = true;
            }
        }
        
        // --- Reassemble the file ---
        // The new file consists of:
        // 1. Everything up to (and including) the marker.
        // 2. The new sorted section.
        // 3. Optionally, an extra blank line (if needed).
        // 4. The remainder of the file.
        vector<string> finalLines;
        for (int i = 0; i < sortedStart; i++)
            finalLines.push_back(lines[i]);
        for (auto &l : newSortedSection)
            finalLines.push_back(l);
        if (needExtraBlank)
            finalLines.push_back(""); // extra blank line added here
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

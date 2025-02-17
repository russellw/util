#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// Count how many leading tabs a string has.
int countLeadingTabs(const string &s) {
    int count = 0;
    for (char ch : s) {
        if (ch == '\t')
            ++count;
        else
            break;
    }
    return count;
}

// Remove leading tabs from a string.
string trimLeadingTabs(const string &s) {
    size_t pos = s.find_first_not_of('\t');
    return (pos == string::npos) ? "" : s.substr(pos);
}

int main(int argc, char *argv[]) {
    bool writeBack = false;
    vector<string> files;

    // Process command-line arguments.
    // Usage: sorter [-w] file1 file2 ...
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-w")
            writeBack = true;
        else
            files.push_back(arg);
    }
    if (files.empty()) {
        cerr << "Usage: " << argv[0] << " [-w] file1 file2 ..." << endl;
        return 1;
    }

    // Process each file.
    for (const auto &filename : files) {
        ifstream inFile(filename);
        if (!inFile) {
            cerr << "Error: could not open file " << filename << endl;
            continue;
        }

        // Read entire file into a vector of strings.
        vector<string> originalLines;
        {
            string line;
            while(getline(inFile, line)) {
                // Normalize: if line ends with '\r', remove it.
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                originalLines.push_back(line);
            }
        }
        inFile.close();

        // We'll build the transformed file in newLines.
        vector<string> newLines;
        size_t i = 0;
        while(i < originalLines.size()) {
            const string &line = originalLines[i];

            // Check for the sort region marker.
            if(line.find("// SORT BLOCKS") != string::npos) {
                // Output the marker line as-is.
                newLines.push_back(line);
                int markerIndent = countLeadingTabs(line);
                ++i; // move to the next line

                // Accumulate the lines inside the sort region until "// END" is found.
                // We will collect blocks and any stray lines.
                vector< vector<string> > blocks;
                vector<string> otherLines;

                while(i < originalLines.size() &&
                      originalLines[i].find("// END") == string::npos) {
                    // A block is expected to start with a line that is at the same indent as the marker
                    // and whose first non-tab character is not '}'.
                    if(countLeadingTabs(originalLines[i]) == markerIndent &&
                       !trimLeadingTabs(originalLines[i]).empty() &&
                       trimLeadingTabs(originalLines[i])[0] != '}') {
                        vector<string> block;
                        // The block header.
                        block.push_back(originalLines[i]);
                        ++i;
                        // Read until we hit the block's closing brace: a line at the same indent level
                        // whose first non-tab character is '}'.
                        while(i < originalLines.size()) {
                            // If this line is at the marker indent, it might be the block's closing brace.
                            if(countLeadingTabs(originalLines[i]) == markerIndent) {
                                string trimmed = trimLeadingTabs(originalLines[i]);
                                if(!trimmed.empty() && trimmed[0] == '}') {
                                    // Add the closing brace and end this block.
                                    block.push_back(originalLines[i]);
                                    ++i;
                                    break;
                                }
                            }
                            // Otherwise, add the line as part of the block.
                            block.push_back(originalLines[i]);
                            ++i;
                        }
                        blocks.push_back(block);
                    } else {
                        // Line is not recognized as a block header; add it verbatim.
                        otherLines.push_back(originalLines[i]);
                        ++i;
                    }
                } // end while (inside sort region)

                // First, output any stray lines that were not part of a block.
                for (const auto &ol : otherLines)
                    newLines.push_back(ol);

                // Sort the blocks alphabetically by their header (ignoring leading tabs).
                sort(blocks.begin(), blocks.end(),
                     [](const vector<string>& a, const vector<string>& b) {
                         string aHeader = a.empty() ? "" : trimLeadingTabs(a[0]);
                         string bHeader = b.empty() ? "" : trimLeadingTabs(b[0]);
                         return aHeader < bHeader;
                     });

                // Output the sorted blocks.
                for (const auto &block : blocks) {
                    for (const auto &bline : block)
                        newLines.push_back(bline);
                }

                // Now, output the "// END" marker if present.
                if(i < originalLines.size()) {
                    newLines.push_back(originalLines[i]); // this is the "// END" line
                    ++i;
                }
            } else {
                // Any line outside a sort region is copied unchanged.
                newLines.push_back(line);
                ++i;
            }
        } // end while over lines

        // Compare newLines with the original.
        bool changed = (newLines != originalLines);

        // If -w was given, overwrite the file only if changes were made.
        if(writeBack) {
            if(changed) {
                ofstream outFile(filename, ios::binary);
                if(!outFile) {
                    cerr << "Error: could not write file " << filename << endl;
                    continue;
                }
                // Write using UNIX line endings.
                for (const auto &outLine : newLines)
                    outFile << outLine << "\n";
                outFile.close();
            }
        } else {
            // Otherwise, print the new content to standard output.
            for (const auto &outLine : newLines)
                cout << outLine << "\n";
        }
    } // end for each file

    return 0;
}

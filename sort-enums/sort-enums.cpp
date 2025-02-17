#include <algorithm>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

// Structure to hold an enum item along with its associated comment lines.
struct EnumItem {
    vector<string> comments; // Each comment line (as read, including the tab)
    string itemLine;         // The actual enum item line (starts with a tab)
    string key;              // The key used for sorting (extracted from the item line)
};

// Extracts the identifier from an enum item line. Assumes the line starts with a tab.
string extractKey(const string &line) {
    // Remove the initial tab
    string s = line.substr(1);
    // Use a regular expression to capture an identifier:
    // an identifier begins with [A-Za-z_] followed by alphanumeric/underscore characters.
    regex re(R"(^\s*([A-Za-z_][A-Za-z0-9_]*))");
    smatch m;
    if (regex_search(s, m, re)) {
        return m[1].str();
    }
    return s; // fallback: return the whole trimmed line if no match
}

int main(int argc, char* argv[]) {
    bool writeInPlace = false;
    vector<string> filenames;

    // Parse command-line options.
    // Usage: enum_sorter [-w] file1 file2 ...
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if(arg == "-w") {
            writeInPlace = true;
        } else {
            filenames.push_back(arg);
        }
    }
    if (filenames.empty()) {
        cerr << "Usage: " << argv[0] << " [-w] <file1> <file2> ...\n";
        return 1;
    }

    // Process each file.
    for (const auto &filename : filenames) {
        ifstream in(filename);
        if (!in) {
            cerr << "Could not open file " << filename << "\n";
            continue;
        }
        // Read entire file into a vector of lines.
        vector<string> lines;
        {
            string line;
            while (getline(in, line)) {
                lines.push_back(line);
            }
        }
        in.close();

        vector<string> outLines;
        size_t i = 0;
        while (i < lines.size()) {
            string cur = lines[i];
            // Look for an enum definition at global scope (line starts with "enum")
            if (cur.compare(0, 4, "enum") == 0) {
                // --- Collect header lines ---
                // The header includes the "enum" line and any subsequent header lines
                // up to (and including) the line that contains the opening brace '{'.
                vector<string> header;
                header.push_back(cur);
                bool foundBrace = (cur.find("{") != string::npos);
                i++;
                while (!foundBrace && i < lines.size()) {
                    // We assume header lines are unindented.
                    if (lines[i].find("{") != string::npos) {
                        header.push_back(lines[i]);
                        foundBrace = true;
                        i++;
                        break;
                    } else {
                        header.push_back(lines[i]);
                        i++;
                    }
                }

                // --- Collect enum body ---
                // According to our simplifying assumption every enum item (and any associated comment)
                // is indented with a single tab. (Any blank lines inside the enum are ignored.)
                vector<string> body;
                while (i < lines.size()) {
                    if (lines[i].empty()) {
                        // Skip blank lines (we don’t preserve them from input).
                        i++;
                        continue;
                    }
                    // If the line starts with a tab, it belongs to the enum body.
                    if (lines[i][0] == '\t')
                        body.push_back(lines[i]);
                    else
                        break; // reached a line that is not part of the enum
                    i++;
                }

                // --- Collect footer lines ---
                // The remainder (typically the closing brace and maybe a semicolon)
                vector<string> footer;
                while (i < lines.size()) {
                    // Stop if we find a line that might start a new enum or other global code.
                    // (We assume footer lines are unindented.)
                    if (!lines[i].empty() && lines[i][0] != '\t') {
                        footer.push_back(lines[i]);
                        i++;
                    } else {
                        break;
                    }
                }

                // --- Group enum body into items ---
                // An enum item may be preceded by one or more comment lines.
                // A comment line is assumed to be one that, after the tab, begins with "//".
                vector<EnumItem> items;
                vector<string> pendingComments;
                for (const auto &bline : body) {
                    // Remove the initial tab and any extra whitespace for checking.
                    string content = bline.substr(1);
                    size_t firstNonSpace = content.find_first_not_of(" \t");
                    string trimmed = (firstNonSpace != string::npos) ? content.substr(firstNonSpace) : "";
                    if (trimmed.find("//") == 0) {
                        // This is a comment line.
                        pendingComments.push_back(bline);
                    } else {
                        // This is the actual enum item.
                        EnumItem item;
                        item.comments = pendingComments;
                        pendingComments.clear();
                        item.itemLine = bline;
                        item.key = extractKey(bline);
                        items.push_back(item);
                    }
                }

                // --- Sort the enum items alphabetically by the extracted key ---
                sort(items.begin(), items.end(), [](const EnumItem &a, const EnumItem &b) {
                    return a.key < b.key;
                });

                // --- Write the modified enum block ---
                // First, output header lines exactly as they appeared.
                for (const auto &h : header)
                    outLines.push_back(h);

                // Then, output the sorted items.
                // When an item has associated comments, we insert a blank line before the first comment
                // (unless one was just output) and a blank line after the item.
                for (const auto &item : items) {
                    if (!item.comments.empty()) {
                        if (outLines.empty() || outLines.back() != "")
                            outLines.push_back(""); // blank line before comment block
                        for (const auto &cmt : item.comments)
                            outLines.push_back(cmt);
                        outLines.push_back(item.itemLine);
                        outLines.push_back(""); // blank line after the item
                    } else {
                        outLines.push_back(item.itemLine);
                    }
                }

                // Finally, output the footer lines.
                for (const auto &f : footer)
                    outLines.push_back(f);
            } else {
                // Not an enum definition line; output it unchanged.
                outLines.push_back(cur);
                i++;
            }
        }

        // Reassemble the output text using UNIX line endings.
        ostringstream oss;
        for (const auto &ol : outLines)
            oss << ol << "\n";
        string output = oss.str();

        if (writeInPlace) {
            // Overwrite the input file.
            ofstream out(filename);
            if (!out) {
                cerr << "Could not write file " << filename << "\n";
                continue;
            }
            out << output;
            out.close();
        } else {
            // Write to standard output.
            cout << output;
        }
    }
    return 0;
}

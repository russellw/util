#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <sstream>

// Structure to represent a case block
struct CaseBlock {
    std::vector<std::string> lines;
    std::string sortKey;
};

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

// Check if a line starts with "case" (after tabs)
bool isCase(const std::string& line) {
    size_t pos = line.find_first_not_of('\t');
    if (pos == std::string::npos) return false;
    
    return line.substr(pos, 5) == "case ";
}

// Extract the sort key from a case line
std::string extractSortKey(const std::string& line) {
    size_t startPos = line.find("case ") + 5;
    std::string key;
    
    // Extract everything up to the colon or comma
    size_t endPos = line.find(':', startPos);
    size_t commaPos = line.find(',', startPos);
    if (commaPos != std::string::npos && (endPos == std::string::npos || commaPos < endPos)) {
        key = line.substr(startPos, commaPos - startPos);
    } else if (endPos != std::string::npos) {
        key = line.substr(startPos, endPos - startPos);
    }
    
    // Trim whitespace
    key.erase(0, key.find_first_not_of(" \t"));
    key.erase(key.find_last_not_of(" \t") + 1);
    
    return key;
}

// Sort cases in switch statements within a file
std::string sortCases(const std::string& content) {
    std::vector<std::string> lines;
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
        // Look for switch statements
        size_t pos = lines[i].find("switch");
        if (pos != std::string::npos && pos == lines[i].find_first_not_of('\t')) {
            // Found a switch statement, add it to result
            result.push_back(lines[i]);
            i++;
            
            int switchIndent = countLeadingTabs(lines[i-1]);
            std::vector<CaseBlock> caseBlocks;
            CaseBlock currentBlock;
            bool inCaseBlock = false;

            
            // Process the switch body
            while (i < lines.size()) {
                line = lines[i];
                int currentIndent = countLeadingTabs(line);
                
                // Check if we've exited the switch statement 
                // (closing brace at same indentation level as the switch)
                if (currentIndent == switchIndent && line.find('}') != std::string::npos) {
                    // End of switch statement
                    if (inCaseBlock) {
                        caseBlocks.push_back(currentBlock);
                    }
                    
                    // Sort the case blocks
                    std::sort(caseBlocks.begin(), caseBlocks.end(), 
                              [](const CaseBlock& a, const CaseBlock& b) {
                                  return a.sortKey < b.sortKey;
                              });
                    
                    // Add sorted case blocks to result
                    for (const auto& block : caseBlocks) {
                        for (const auto& blockLine : block.lines) {
                            result.push_back(blockLine);
                        }
                    }
                    
                    // Add the closing brace
                    result.push_back(line);
                    i++;
                    break;
                }
                
                // We don't track brace balance - using indentation level only
                
                // Check if this is a new case
                if (isCase(line) && currentIndent == switchIndent + 1) {
                    if (inCaseBlock) {
                        caseBlocks.push_back(currentBlock);
                    }
                    
                    // Start a new case block
                    currentBlock = CaseBlock();
                    currentBlock.sortKey = extractSortKey(line);
                    currentBlock.lines.push_back(line);
                    inCaseBlock = true;
                } else if (inCaseBlock) {
                    // Detect end of case block by finding another case or indentation level
                    // A case ends when we find another case at the same level,
                    // or when we find a line with the same indentation as the switch
                    // (unless it's a blank line)
                    if ((isCase(line) && currentIndent == switchIndent + 1) || 
                        (currentIndent == switchIndent && !line.empty() && line.find_first_not_of("\t ") != std::string::npos)) {
                        caseBlocks.push_back(currentBlock);
                        
                        // If this is a new case, start a new block
                        if (isCase(line)) {
                            currentBlock = CaseBlock();
                            currentBlock.sortKey = extractSortKey(line);
                            currentBlock.lines.push_back(line);
                        } else {
                            // This isn't a case, so we're not in a case block anymore
                            inCaseBlock = false;
                            result.push_back(line);
                        }
                    } else {
                        // Continue the current case block
                        currentBlock.lines.push_back(line);
                    }
                } else {
                    // Not in a case block yet (e.g., the opening brace of the switch)
                    result.push_back(line);
                }
                
                i++;
            }
        } else {
            // Not a switch statement, just copy the line
            result.push_back(lines[i]);
            i++;
        }
    }
    
    // Join lines with UNIX line endings
    std::ostringstream oss;
    for (size_t i = 0; i < result.size(); ++i) {
        oss << result[i];
        if (i < result.size() - 1) {
            oss << '\n';
        }
    }
    
    return oss.str();
}

int main(int argc, char* argv[]) {
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
            std::string sortedContent = sortCases(content);
            
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
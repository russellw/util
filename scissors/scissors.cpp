#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <regex>

int main(int argc, char* argv[]) {
    // Check if an input file was provided
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " input_file" << std::endl;
        return 1;
    }

    // Open the input file
    std::ifstream input_file(argv[1]);
    if (!input_file) {
        std::cerr << "Error: Could not open input file " << argv[1] << std::endl;
        return 1;
    }

    // Read the file into a vector of lines
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input_file, line)) {
        lines.push_back(line);
    }
    input_file.close();

    // Regular expression for marker lines: -- followed by a letter or digit
    std::regex marker_pattern("^--([a-zA-Z0-9].*)$");
    std::smatch match;

    // Process the file
    std::vector<std::string> current_content;

    for (const auto& line : lines) {
        if (std::regex_match(line, match, marker_pattern)) {
            // Found a marker line, extract the output file name
            std::string output_file_name = match[1];
            
            // Write accumulated content to the output file
            if (!current_content.empty()) {
                std::ofstream output(output_file_name, std::ios::binary);
                if (!output) {
                    std::cerr << "Error: Could not open output file " << output_file_name << std::endl;
                    return 1;
                }
                
                for (const auto& content_line : current_content) {
                    output << content_line << '\n';
                }
                output.close();
            }
            
            // Clear content for the next section
            current_content.clear();
        } else {
            // Regular content line, add to current content
            current_content.push_back(line);
        }
    }

    return 0;
}
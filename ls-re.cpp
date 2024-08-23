#include <iostream>
#include <filesystem>
#include <regex>
#include <vector>
#include <string>

namespace fs = std::filesystem;

void search_directory(const fs::path& dir, const std::regex& pattern, std::vector<fs::path>& matches) {
    for (const auto& entry : fs::recursive_directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            if (std::regex_match(entry.path().filename().string(), pattern)) {
                matches.push_back(entry.path());
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <directory> <regex>" << std::endl;
        return 1;
    }

    fs::path directory = argv[1];
    std::string regex_pattern = argv[2];

    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        std::cerr << "Error: Directory " << directory << " does not exist or is not a directory." << std::endl;
        return 1;
    }

    std::regex pattern(regex_pattern);
    std::vector<fs::path> matches;

    try {
        search_directory(directory, pattern, matches);
    } catch (const std::regex_error& e) {
        std::cerr << "Error: Invalid regular expression. " << e.what() << std::endl;
        return 1;
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error: Filesystem error. " << e.what() << std::endl;
        return 1;
    }

    for (const auto& match : matches) {
        std::cout << match << std::endl;
    }

    return 0;
}

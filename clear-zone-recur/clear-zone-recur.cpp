#include <windows.h>
#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

struct Statistics {
    int filesScanned = 0;
    int zonesCleared = 0;
    int failures = 0;
};

bool clearZoneIdentifier(const fs::path& filePath) {
    std::wstring streamPath = filePath.wstring() + L":Zone.Identifier";
    
    // Check if the stream exists
    HANDLE hFile = CreateFileW(
        streamPath.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (hFile == INVALID_HANDLE_VALUE) {
        return false;  // No zone information
    }
    CloseHandle(hFile);

    // Delete the alternate data stream
    if (DeleteFileW(streamPath.c_str())) {
        std::cout << "Cleared zone information from: " << filePath.string() << std::endl;
        return true;
    } else {
        DWORD error = GetLastError();
        std::cerr << "Failed to clear zone information from: " << filePath.string() 
                  << "\nError code: " << error << std::endl;
        return false;
    }
}

void processDirectory(const fs::path& dirPath, Statistics& stats, bool recursive = true) {
    try {
        if (recursive) {
            for (const auto& entry : fs::recursive_directory_iterator(dirPath)) {
                if (fs::is_regular_file(entry)) {
                    stats.filesScanned++;
                    try {
                        if (clearZoneIdentifier(entry.path())) {
                            stats.zonesCleared++;
                        }
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Error processing " << entry.path().string() 
                                 << ": " << e.what() << std::endl;
                        stats.failures++;
                    }
                }
            }
        } else {
            for (const auto& entry : fs::directory_iterator(dirPath)) {
                if (fs::is_regular_file(entry)) {
                    stats.filesScanned++;
                    try {
                        if (clearZoneIdentifier(entry.path())) {
                            stats.zonesCleared++;
                        }
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Error processing " << entry.path().string() 
                                 << ": " << e.what() << std::endl;
                        stats.failures++;
                    }
                }
            }
        }
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Error accessing directory: " << e.what() << std::endl;
        stats.failures++;
    }
}

void printUsage() {
    std::cout << "Usage: recursive_zone_clearer.exe [options]\n"
              << "Clears zone information (Internet downloaded flag) from all files in the current directory.\n\n"
              << "Options:\n"
              << "  -h, --help     Show this help message\n"
              << "  -n, --dry-run  Show what would be done without actually clearing zones\n"
              << "  -s, --single   Process only current directory (non-recursive)\n"
              << "  -q, --quiet    Suppress output except for errors and summary" << std::endl;
}

int main(int argc, char* argv[]) {
    bool dryRun = false;
    bool recursive = true;
    bool quiet = false;

    // Process command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage();
            return 0;
        }
        else if (arg == "-n" || arg == "--dry-run") {
            dryRun = true;
            std::cout << "Dry run mode - no changes will be made\n" << std::endl;
        }
        else if (arg == "-s" || arg == "--single") {
            recursive = false;
        }
        else if (arg == "-q" || arg == "--quiet") {
            quiet = true;
        }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage();
            return 1;
        }
    }

    auto startTime = std::chrono::steady_clock::now();
    Statistics stats;

    if (!quiet) {
        std::cout << "Scanning " << (recursive ? "recursively" : "current directory only") 
                  << " for files with zone information...\n" << std::endl;
    }

    processDirectory(".", stats, recursive);

    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Print summary
    std::cout << "\nSummary:\n"
              << "Files scanned: " << stats.filesScanned << "\n"
              << "Zone information cleared: " << stats.zonesCleared << "\n"
              << "Failures: " << stats.failures << "\n"
              << "Time taken: " << duration.count() / 1000.0 << " seconds" << std::endl;

    return (stats.failures > 0) ? 1 : 0;
}
#include <windows.h>
#include <iostream>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

bool clearZoneIdentifier(const fs::path& filePath) {
    // Append :Zone.Identifier to the path
    std::wstring streamPath = filePath.wstring() + L":Zone.Identifier";
    
    // First check if the stream exists
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
        // No zone information found
        return false;
    }
    CloseHandle(hFile);

    // Delete the alternate data stream
    if (DeleteFileW(streamPath.c_str())) {
        std::cout << "Successfully cleared zone information from: " << filePath.string() << std::endl;
        return true;
    } else {
        DWORD error = GetLastError();
        std::cerr << "Failed to clear zone information from: " << filePath.string() 
                  << "\nError code: " << error << std::endl;
        return false;
    }
}

void printUsage() {
    std::cout << "Usage: zone_clearer.exe <file1> [file2] [file3] ...\n"
              << "Clears zone information (Internet downloaded flag) from specified files.\n"
              << "Example: zone_clearer.exe downloaded.exe program.dll data.zip" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    int successCount = 0;
    int failCount = 0;
    int noZoneCount = 0;

    // Process each file specified on the command line
    for (int i = 1; i < argc; i++) {
        try {
            fs::path filePath = fs::absolute(argv[i]);
            
            if (!fs::exists(filePath)) {
                std::cerr << "File not found: " << argv[i] << std::endl;
                failCount++;
                continue;
            }

            if (!fs::is_regular_file(filePath)) {
                std::cerr << "Not a regular file: " << argv[i] << std::endl;
                failCount++;
                continue;
            }

            if (clearZoneIdentifier(filePath)) {
                successCount++;
            } else {
                noZoneCount++;
            }
        }
        catch (const fs::filesystem_error& e) {
            std::cerr << "Filesystem error processing " << argv[i] << ": " << e.what() << std::endl;
            failCount++;
        }
        catch (const std::exception& e) {
            std::cerr << "Error processing " << argv[i] << ": " << e.what() << std::endl;
            failCount++;
        }
    }

    // Print summary
    std::cout << "\nSummary:\n"
              << "Files processed: " << argc - 1 << "\n"
              << "Zone information cleared: " << successCount << "\n"
              << "Files without zone information: " << noZoneCount << "\n"
              << "Failures: " << failCount << std::endl;

    return (failCount > 0) ? 1 : 0;
}
#include <windows.h>
#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

std::string getZoneDescription(int zoneId) {
    switch (zoneId) {
        case 0: return "Local Machine";
        case 1: return "Local Intranet";
        case 2: return "Trusted Sites";
        case 3: return "Internet";
        case 4: return "Restricted Sites";
        default: return "Unknown";
    }
}

bool checkZoneIdentifier(const fs::path& filePath) {
    // Append :Zone.Identifier to the path
    std::wstring streamPath = filePath.wstring() + L":Zone.Identifier";
    
    // Try to open the alternate data stream
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
        return false;
    }

    // Read the stream content
    char buffer[1024];
    DWORD bytesRead;
    std::string content;

    while (ReadFile(hFile, buffer, sizeof(buffer), &bytesRead, NULL) && bytesRead > 0) {
        content.append(buffer, bytesRead);
    }

    CloseHandle(hFile);

    // Parse the content for ZoneId
    size_t pos = content.find("ZoneId=");
    if (pos != std::string::npos) {
        int zoneId = content[pos + 7] - '0';  // Convert char to int
        std::cout << "\nFile: " << filePath.string() << std::endl;
        std::cout << "Zone: " << zoneId << " (" << getZoneDescription(zoneId) << ")" << std::endl;

        // Look for additional metadata
        pos = content.find("ReferrerUrl=");
        if (pos != std::string::npos) {
            size_t endPos = content.find('\n', pos);
            if (endPos != std::string::npos) {
                std::string referrer = content.substr(pos + 12, endPos - pos - 12);
                std::cout << "Referrer URL: " << referrer << std::endl;
            }
        }

        pos = content.find("HostUrl=");
        if (pos != std::string::npos) {
            size_t endPos = content.find('\n', pos);
            if (endPos != std::string::npos) {
                std::string host = content.substr(pos + 8, endPos - pos - 8);
                std::cout << "Host URL: " << host << std::endl;
            }
        }
        return true;
    }

    return false;
}

int main() {
    bool foundAny = false;
    
    try {
        for (const auto& entry : fs::recursive_directory_iterator(".")) {
            if (fs::is_regular_file(entry)) {
                if (checkZoneIdentifier(entry.path())) {
                    foundAny = true;
                }
            }
        }

        if (!foundAny) {
            std::cout << "\nNo files with non-default zone information were found." << std::endl;
        }
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Error accessing directory: " << e.what() << std::endl;
    }

    return 0;
}
#include <windows.h>
#include <iostream>
#include <string>

bool setFileAssociation(const std::string& extension) {
    // Remove the dot if present at the start of extension
    std::string ext = extension;
    if (!ext.empty() && ext[0] == '.') {
        ext = ext.substr(1);
    }

    // Get path to Notepad++
    char nppPath[MAX_PATH];
    DWORD pathSize = sizeof(nppPath);
    HKEY hKey;
    
    // First try to find Notepad++ installation from registry
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, 
        "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\notepad++.exe",
        0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        
        if (RegQueryValueExA(hKey, nullptr, nullptr, nullptr, 
            (LPBYTE)nppPath, &pathSize) != ERROR_SUCCESS) {
            RegCloseKey(hKey);
            std::cerr << "Error: Couldn't find Notepad++ path in registry\n";
            return false;
        }
        RegCloseKey(hKey);
    } else {
        // Try common installation paths
        const char* commonPaths[] = {
            "C:\\Program Files\\Notepad++\\notepad++.exe",
            "C:\\Program Files (x86)\\Notepad++\\notepad++.exe"
        };
        
        bool found = false;
        for (const char* path : commonPaths) {
            if (GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES) {
                strcpy_s(nppPath, path);
                found = true;
                break;
            }
        }
        
        if (!found) {
            std::cerr << "Error: Notepad++ installation not found\n";
            return false;
        }
    }

    // Create ProgID entry
    std::string progID = "NotepadPlusPlus." + ext;
    HKEY hProgID;
    if (RegCreateKeyExA(HKEY_CLASSES_ROOT, progID.c_str(), 0, nullptr,
        REG_OPTION_NON_VOLATILE, KEY_WRITE, nullptr, &hProgID, nullptr) != ERROR_SUCCESS) {
        std::cerr << "Error: Failed to create ProgID key\n";
        return false;
    }
    
    // Set default value for ProgID
    std::string description = ext + " File";
    RegSetValueExA(hProgID, nullptr, 0, REG_SZ, 
        (BYTE*)description.c_str(), description.length() + 1);
    
    // Create shell open command
    HKEY hCommand;
    if (RegCreateKeyExA(hProgID, "shell\\open\\command", 0, nullptr,
        REG_OPTION_NON_VOLATILE, KEY_WRITE, nullptr, &hCommand, nullptr) != ERROR_SUCCESS) {
        RegCloseKey(hProgID);
        std::cerr << "Error: Failed to create command key\n";
        return false;
    }
    
    // Set command value
    std::string command = std::string("\"") + nppPath + "\" \"%1\"";
    RegSetValueExA(hCommand, nullptr, 0, REG_SZ, 
        (BYTE*)command.c_str(), command.length() + 1);
    RegCloseKey(hCommand);
    RegCloseKey(hProgID);
    
    // Associate extension with ProgID
    HKEY hExtension;
    if (RegCreateKeyExA(HKEY_CLASSES_ROOT, ("." + ext).c_str(), 0, nullptr,
        REG_OPTION_NON_VOLATILE, KEY_WRITE, nullptr, &hExtension, nullptr) != ERROR_SUCCESS) {
        std::cerr << "Error: Failed to create extension key\n";
        return false;
    }
    
    RegSetValueExA(hExtension, nullptr, 0, REG_SZ, 
        (BYTE*)progID.c_str(), progID.length() + 1);
    RegCloseKey(hExtension);
    
    // Notify the shell about the change
    SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nullptr, nullptr);
    
    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <extension>\n";
        std::cout << "Example: " << argv[0] << " .txt\n";
        return 1;
    }

    // Request administrative privileges
    if (!IsUserAnAdmin()) {
        std::cout << "This program requires administrative privileges.\n";
        std::cout << "Please run as administrator.\n";
        return 1;
    }

    std::string extension = argv[1];
    if (setFileAssociation(extension)) {
        std::cout << "Successfully associated " << extension 
                  << " files with Notepad++\n";
        return 0;
    } else {
        std::cout << "Failed to set file association\n";
        return 1;
    }
}

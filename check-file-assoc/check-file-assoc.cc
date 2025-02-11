#include <windows.h>
#include <iostream>
#include <string>
#include <vector>

// Helper function to read a registry value
std::wstring ReadRegistryValue(HKEY hKey, const std::wstring& subKey, const std::wstring& value) {
    HKEY hSubKey;
    if (RegOpenKeyExW(hKey, subKey.c_str(), 0, KEY_READ, &hSubKey) != ERROR_SUCCESS) {
        return L"";
    }

    WCHAR buffer[MAX_PATH];
    DWORD bufferSize = sizeof(buffer);
    DWORD type = REG_SZ;

    if (RegQueryValueExW(hSubKey, value.c_str(), nullptr, &type, 
        reinterpret_cast<LPBYTE>(buffer), &bufferSize) != ERROR_SUCCESS) {
        RegCloseKey(hSubKey);
        return L"";
    }

    RegCloseKey(hSubKey);
    return std::wstring(buffer);
}

// Helper function to check if a ProgID has AppUserModelID
bool HasAppUserModelID(const std::wstring& progID) {
    std::wstring subKey = progID + L"\\Application";
    HKEY hKey;
    if (RegOpenKeyExW(HKEY_CLASSES_ROOT, subKey.c_str(), 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
        RegCloseKey(hKey);
        return true;
    }
    return false;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <file-extension>\n";
        std::cout << "Example: " << argv[0] << " .txt\n";
        return 1;
    }

    // Convert extension to wide string and ensure it starts with a dot
    std::string ext = argv[1];
    if (ext[0] != '.') {
        ext = "." + ext;
    }
    std::wstring extension(ext.begin(), ext.end());

    // Get the ProgID associated with the extension
    std::wstring progID = ReadRegistryValue(HKEY_CLASSES_ROOT, extension, L"");
    if (progID.empty()) {
        std::wcout << L"No association found for " << extension << std::endl;
        return 1;
    }

    std::wcout << L"File Extension: " << extension << L"\n";
    std::wcout << L"ProgID: " << progID << L"\n\n";

    // Get the default command
    std::wstring command = ReadRegistryValue(HKEY_CLASSES_ROOT, 
        progID + L"\\shell\\open\\command", L"");
    if (!command.empty()) {
        std::wcout << L"Default Open Command:\n" << command << L"\n\n";
    }

    // Check for potential issues
    std::wcout << L"Additional Information:\n";
    
    // Check if there's a user choice override
    std::wstring userChoice = ReadRegistryValue(HKEY_CURRENT_USER, 
        L"Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\FileExts\\" + 
        extension + L"\\UserChoice", L"ProgId");
    if (!userChoice.empty()) {
        std::wcout << L"- User Choice Override exists: " << userChoice << L"\n";
        if (userChoice != progID) {
            std::wcout << L"  Warning: User choice differs from default association\n";
        }
    }

    // Check for App Paths registration
    std::wstring appPaths = ReadRegistryValue(HKEY_LOCAL_MACHINE,
        L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths", progID);
    if (!appPaths.empty()) {
        std::wcout << L"- Registered in App Paths\n";
    }

    // Check for AppUserModelID
    if (HasAppUserModelID(progID)) {
        std::wcout << L"- Has AppUserModelID registration\n";
    }

    // Check for OpenWithProgIds
    HKEY hOpenWith;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, 
        (L"Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\FileExts\\" + 
         extension + L"\\OpenWithProgids").c_str(), 
        0, KEY_READ, &hOpenWith) == ERROR_SUCCESS) {
        std::wcout << L"- Has OpenWithProgIds entries\n";
        RegCloseKey(hOpenWith);
    }

    // Check for potential policy restrictions
    HKEY hPolicy;
    if (RegOpenKeyExW(HKEY_CURRENT_USER,
        L"Software\\Microsoft\\Windows\\CurrentVersion\\Policies\\Explorer",
        0, KEY_READ, &hPolicy) == ERROR_SUCCESS) {
        DWORD noFileAssociate = 0;
        DWORD size = sizeof(DWORD);
        if (RegQueryValueExW(hPolicy, L"NoFileAssociate", nullptr, nullptr,
            reinterpret_cast<LPBYTE>(&noFileAssociate), &size) == ERROR_SUCCESS) {
            if (noFileAssociate) {
                std::wcout << L"! Warning: File association changes are restricted by policy\n";
            }
        }
        RegCloseKey(hPolicy);
    }

    return 0;
}

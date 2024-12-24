#include <windows.h>
#include <iostream>
#include <string>
#include <sstream>

bool GetWindowInfo(HWND hwnd, std::string &title, std::string &className, DWORD &processID) {
    // Get the window title
    char windowTitle[256];
    if (GetWindowTextA(hwnd, windowTitle, sizeof(windowTitle)) == 0) {
        return false; // Skip windows with no title
    }
    title = windowTitle;

    // Get the class name
    char windowClass[256];
    if (GetClassNameA(hwnd, windowClass, sizeof(windowClass)) == 0) {
        return false; // Skip windows with no class name
    }
    className = windowClass;

    // Get the process ID
    GetWindowThreadProcessId(hwnd, &processID);

    return true;
}

BOOL CALLBACK EnumWindowsProc(HWND hwnd, LPARAM lParam) {
    std::ostringstream *output = reinterpret_cast<std::ostringstream *>(lParam);

    if (IsWindowVisible(hwnd)) {
        std::string title, className;
        DWORD processID;
        if (GetWindowInfo(hwnd, title, className, processID)) {
            // Escape any quotes or commas in the output
            auto escape_csv = [](const std::string &input) -> std::string {
                if (input.find(',') != std::string::npos || input.find('"') != std::string::npos) {
                    std::string escaped = "\"";
                    for (char c : input) {
                        if (c == '"') escaped += "\"\"";
                        else escaped += c;
                    }
                    escaped += "\"";
                    return escaped;
                }
                return input;
            };

            *output << processID << ","
                    << escape_csv(className) << ","
                    << escape_csv(title) << "\n";
        }
    }
    return TRUE;
}

int main() {
    std::ostringstream output;

    if (EnumWindows(EnumWindowsProc, reinterpret_cast<LPARAM>(&output))) {
        std::cout << output.str();
    } else {
        std::cerr << "Failed to enumerate windows.\n";
        return 1;
    }

    return 0;
}

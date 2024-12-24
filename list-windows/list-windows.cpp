#include <windows.h>
#include <iostream>
#include <iomanip>
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
            // Adjust columns for fixed-width output
            *output << std::setw(10) << processID << " " // Process ID: width 10
                    << std::setw(30) << std::left << className << " " // Class Name: width 30
                    << title.substr(0, 50) << "\n"; // Title: truncate to 50 chars if too long
        }
    }
    return TRUE;
}

int main() {
    std::ostringstream output;

    // Print header row
    output << std::setw(10) << "Proc ID" << " "
           << std::setw(30) << std::left << "Class Name" << " "
           << "Title" << "\n";
    output << std::string(80, '-') << "\n"; // Separator line

    if (EnumWindows(EnumWindowsProc, reinterpret_cast<LPARAM>(&output))) {
        std::cout << output.str();
    } else {
        std::cerr << "Failed to enumerate windows.\n";
        return 1;
    }

    return 0;
}

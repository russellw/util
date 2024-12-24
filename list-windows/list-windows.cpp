#include <iostream>
#include <windows.h>
#include <string>
#include <vector>
#include <memory>

struct WindowInfo {
    HWND hwnd;
    std::string title;
    std::string className;
    DWORD processId;
    bool isVisible;
};

BOOL CALLBACK EnumWindowsProc(HWND hwnd, LPARAM lParam) {
    // Get window title
    char title[256];
    GetWindowTextA(hwnd, title, sizeof(title));

    // Get window class name
    char className[256];
    GetClassNameA(hwnd, className, sizeof(className));

    // Get process ID
    DWORD processId;
    GetWindowThreadProcessId(hwnd, &processId);

    // Check visibility of the window
    bool isVisible = IsWindowVisible(hwnd);

    // Store the window info in the vector
    std::vector<WindowInfo>* windows = reinterpret_cast<std::vector<WindowInfo>*>(lParam);
    windows->emplace_back(WindowInfo{ hwnd, title, className, processId, isVisible });

    return TRUE;
}

void PrintWindowInfo(const std::vector<WindowInfo>& windows) {
    std::cout << "List of currently open windows:" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    for (const auto& win : windows) {
        std::cout << "HWND: " << win.hwnd << std::endl;
        std::cout << "Title: \"" << win.title << "\"" << std::endl;
        std::cout << "Class Name: " << win.className << std::endl;
        std::cout << "Process ID: " << win.processId << std::endl;
        std::cout << "Visible: " << (win.isVisible ? "Yes" : "No") << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }
}

int main() {
    std::vector<WindowInfo> windows;

    // Enumerate all top-level windows
    EnumWindows(EnumWindowsProc, reinterpret_cast<LPARAM>(&windows));

    // Print the collected information
    PrintWindowInfo(windows);

    return 0;
}

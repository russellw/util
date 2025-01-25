#include <iostream>
#include <string>
#include <windows.h>
#include <sstream>

void PrintUsage() {
    std::cout << "Usage: ResizeWindow.exe <WindowTitle|ProcessID> <Width> <Height>" << std::endl;
}

HWND FindWindowByTitle(const std::string& title) {
    return FindWindowA(NULL, title.c_str());
}

HWND FindWindowByProcessID(DWORD processID) {
    HWND hwnd = NULL;
    do {
        hwnd = FindWindowExA(NULL, hwnd, NULL, NULL);
        DWORD pid = 0;
        GetWindowThreadProcessId(hwnd, &pid);
        if (pid == processID) {
            char className[256];
            if (IsWindowVisible(hwnd) && GetClassNameA(hwnd, className, sizeof(className))) {
                return hwnd;
            }
        }
    } while (hwnd != NULL);
    return NULL;
}

bool IsInteger(const std::string& str) {
    std::istringstream iss(str);
    int num;
    return (iss >> num) && (iss.eof());
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        PrintUsage();
        return 1;
    }

    std::string windowArg = argv[1];
    int width = std::stoi(argv[2]);
    int height = std::stoi(argv[3]);

    HWND hwnd = NULL;

    if (IsInteger(windowArg)) {
        DWORD processID = std::stoul(windowArg);
        hwnd = FindWindowByProcessID(processID);
        if (!hwnd) {
            std::cerr << "Error: Could not find a visible window associated with process ID " << processID << std::endl;
            return 1;
        }
    } else {
        hwnd = FindWindowByTitle(windowArg);
        if (!hwnd) {
            std::cerr << "Error: Could not find a window with title \"" << windowArg << "\"" << std::endl;
            return 1;
        }
    }

    RECT rect;
    if (!GetWindowRect(hwnd, &rect)) {
        std::cerr << "Error: Could not retrieve window rectangle" << std::endl;
        return 1;
    }

    int x = rect.left;
    int y = rect.top;

    if (!MoveWindow(hwnd, x, y, width, height, TRUE)) {
        std::cerr << "Error: Could not resize the window" << std::endl;
        return 1;
    }

    std::cout << "Window resized successfully!" << std::endl;
    return 0;
}

#include <iostream>
#include <string>
#include <windows.h>

void PrintUsage() {
    std::cout << "Usage: ResizeWindow.exe <WindowTitle> <Width> <Height>" << std::endl;
}

HWND FindWindowByTitle(const std::string& title) {
    return FindWindowA(NULL, title.c_str());
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        PrintUsage();
        return 1;
    }

    std::string windowTitle = argv[1];
    int width = std::stoi(argv[2]);
    int height = std::stoi(argv[3]);

    HWND hwnd = FindWindowByTitle(windowTitle);
    if (!hwnd) {
        std::cerr << "Error: Could not find a window with title \"" << windowTitle << "\"" << std::endl;
        return 1;
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

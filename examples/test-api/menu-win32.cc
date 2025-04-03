#include <windows.h>

// Function declarations for window procedure and commands
LRESULT CALLBACK WindowProcedure(HWND, UINT, WPARAM, LPARAM);
void AddMenus(HWND);

// Global variables
HMENU hMenu;

int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrevInst, LPSTR args, int ncmdshow) {
    WNDCLASSW wc = {0};

    wc.hbrBackground = (HBRUSH)COLOR_WINDOW;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hInstance = hInst;
    wc.lpszClassName = L"myWindowClass";
    wc.lpfnWndProc = WindowProcedure;

    if (!RegisterClassW(&wc)) {
        return -1;
    }

    CreateWindowW(L"myWindowClass", L"Simple GUI with Menu Bar", WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                  100, 100, 500, 300, NULL, NULL, NULL, NULL);

    MSG msg = {0};
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}

LRESULT CALLBACK WindowProcedure(HWND hWnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
        case WM_COMMAND:
            switch(wp) {
                case 1:
                    MessageBoxW(hWnd, L"New File created!", L"Info", MB_OK);
                    break;
                case 2:
                    MessageBoxW(hWnd, L"About This Program", L"About", MB_OK);
                    break;
            }
            break;
        case WM_CREATE:
            AddMenus(hWnd);
            break;
        case WM_DESTROY:
            PostQuitMessage(0);
            break;
        default:
            return DefWindowProcW(hWnd, msg, wp, lp);
    }
    return 0;
}

void AddMenus(HWND hWnd) {
    hMenu = CreateMenu();

    HMENU hFileMenu = CreateMenu();
    HMENU hSubMenu = CreateMenu();

    AppendMenu(hSubMenu, MF_STRING, 1, L"New");

    AppendMenu(hFileMenu, MF_STRING, 1, L"New");
    AppendMenu(hFileMenu, MF_STRING | MF_POPUP, (UINT_PTR)hSubMenu, L"Open");

    AppendMenu(hMenu, MF_POPUP, (UINT_PTR)hFileMenu, L"File");

    HMENU hHelpMenu = CreateMenu();
    AppendMenu(hHelpMenu, MF_STRING, 2, L"About");

    AppendMenu(hMenu, MF_POPUP, (UINT_PTR)hHelpMenu, L"Help");

    SetMenu(hWnd, hMenu);
}

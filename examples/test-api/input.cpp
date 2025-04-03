#include <windows.h>
#include <string>

#define IDC_TEXTBOX 101
#define IDC_BUTTON 102

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

void Submit(HWND hwnd)
{
    // Get text from the text box
    HWND hEdit = GetDlgItem(hwnd, IDC_TEXTBOX);
    int len = GetWindowTextLength(hEdit);
    if (len > 0)
    {
        char* buffer = new char[len + 1];
        GetWindowText(hEdit, buffer, len + 1);

        // Copy text to clipboard
        if (OpenClipboard(hwnd))
        {
            EmptyClipboard();
            HGLOBAL hglbCopy = GlobalAlloc(GMEM_MOVEABLE, (len + 1) * sizeof(char));
            if (hglbCopy != NULL)
            {
                LPSTR lpstrCopy = (LPSTR)GlobalLock(hglbCopy);
                memcpy(lpstrCopy, buffer, len + 1);
                GlobalUnlock(hglbCopy);
                SetClipboardData(CF_TEXT, hglbCopy);
            }
            CloseClipboard();
        }

        MessageBox(hwnd, "Text copied to clipboard", "Info", MB_OK);

        // Simulate Alt+Tab
        keybd_event(VK_MENU, 0, 0, 0);
        keybd_event(VK_TAB, 0, 0, 0);
        keybd_event(VK_TAB, 0, KEYEVENTF_KEYUP, 0);
        keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0);

        delete[] buffer;
    }
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow)
{
    const wchar_t CLASS_NAME[] = L"Sample Window Class";

    WNDCLASS wc = {};

    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;

    RegisterClass(&wc);

    HWND hwnd = CreateWindowEx(
        0,
        CLASS_NAME,
        L"Paragraph Input Box",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 500, 350,
        NULL,
        NULL,
        hInstance,
        NULL
    );

    if (hwnd == NULL)
    {
        return 0;
    }

    ShowWindow(hwnd, nCmdShow);

    MSG msg = {};
    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_CREATE:
    {
        CreateWindowEx(0, L"STATIC", L"Please enter a paragraph of text:", WS_VISIBLE | WS_CHILD,
                       10, 10, 400, 20, hwnd, NULL, NULL, NULL);

        CreateWindowEx(WS_EX_CLIENTEDGE, L"EDIT", L"", WS_CHILD | WS_VISIBLE | ES_MULTILINE | ES_AUTOVSCROLL | ES_LEFT,
                       10, 40, 460, 200, hwnd, (HMENU)IDC_TEXTBOX, GetModuleHandle(NULL), NULL);

        CreateWindowEx(0, L"BUTTON", L"Submit", WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
                       10, 250, 75, 23, hwnd, (HMENU)IDC_BUTTON, GetModuleHandle(NULL), NULL);
    }
    break;

    case WM_COMMAND:
        if (LOWORD(wParam) == IDC_BUTTON)
        {
            Submit(hwnd);
        }
        break;

    case WM_KEYDOWN:
        if (wParam == VK_F12)
        {
            Submit(hwnd);
        }
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;

    case WM_CLOSE:
        DestroyWindow(hwnd);
        return 0;
    }

    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

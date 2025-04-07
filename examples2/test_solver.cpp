#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <vector>
using std::vector;

	#include<windows.h>

	char buf[0x10000];

static void win_die(char *prefix) {
  char *msg;
  FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS,
                0, GetLastError(), 0, (LPTSTR)&msg, 0, 0);
  printf("%s: %s", prefix, msg);
  exit(1);
}

void test(char*file){
  SECURITY_ATTRIBUTES sa={0};
  sa.bInheritHandle = 1;
  sa.lpSecurityDescriptor = 0;
  sa.nLength = sizeof sa;

  HANDLE child_stdout_read;
  HANDLE child_stdout_write;
  if (!CreatePipe(&child_stdout_read, &child_stdout_write, &sa, 0))
    win_die("CreatePipe");
  if (!SetHandleInformation(child_stdout_read, HANDLE_FLAG_INHERIT, 0))
    win_die("SetHandleInformation");

  STARTUPINFO si;
  ZeroMemory(&si, sizeof si);
  si.cb = sizeof si;
  si.hStdInput = INVALID_HANDLE_VALUE;
  si.hStdOutput = child_stdout_write;
  si.hStdError = INVALID_HANDLE_VALUE;

  PROCESS_INFORMATION pi;
  ZeroMemory(&pi, sizeof pi);

  sprintf(buf, "testc \"%s\"", file);
  if (!CreateProcess(0, buf, 0, 0, 1, 0, 0, 0, &si, &pi))
    win_die("CreateProcess");
  WaitForSingleObject(pi.hProcess, INFINITE);

  DWORD exit_code;
  GetExitCodeProcess(pi.hProcess, &exit_code);

  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  CloseHandle(child_stdout_write);

  if (exit_code)
    exit(1);

  DWORD n;
  if (!ReadFile(child_stdout_read, buf, 0x10000, &n, 0))
    win_die("ReadFile");
  fwrite(buf, n, 1, stdout);
  putchar('\n');
}

int main(int argc, char **argv) {
	test("a.p");
	return 0;
	}

#include "src/ayane.h"
#include <windows.h>

static void win_die(char *prefix) {
  char *msg;
  FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS,
                0, GetLastError(), 0, (LPTSTR)&msg, 0, 0);
  printf("%s: %s", prefix, msg);
  exit(1);
}

static FILE *xfopen(char *filename, char *mode) {
  FILE *f = fopen(filename, mode);
  if (!f) {
    perror(filename);
    exit(1);
  }
  return f;
}

static char *xstrdup(char *s) {
  s = strdup(s);
  if (!s)
    perror("strdup");
  return s;
}

static void test(void) {
  DWORD a = GetFileAttributes(file);
  if (a == INVALID_FILE_ATTRIBUTES)
    win_die(file);

  // avoid cycles
  if (a & FILE_ATTRIBUTE_REPARSE_POINT)
    return;

  // directory
  if (a & FILE_ATTRIBUTE_DIRECTORY) {
    WIN32_FIND_DATA f;
    char *dir = file;
    sprintf(buf, "%s/*", dir);
    HANDLE h = FindFirstFile(buf, &f);
    if (h == INVALID_HANDLE_VALUE)
      win_die(buf);
    do {
      if (*f.cFileName == '.')
        continue;
      sprintf(buf, "%s/%s", dir, f.cFileName);
      file = xstrdup(buf);
      test();
    } while (FindNextFile(h, &f));
    FindClose(h);
    return;
  }

  // language
  char *extension = strrchr(file, '.');
  if (!extension)
    return;
  if (!strcmp(extension, ".cnf")) {
  } else if (!strcmp(extension, ".p")) {
    if (strpbrk(file, "=^_"))
      return;
  } else
    return;

  // regular file
  printf("%40s ", file);

  SECURITY_ATTRIBUTES sa;
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

  sprintf(buf, "ayane \"%s\"", file);
  if (!CreateProcess(0, buf, 0, 0, 1, 0, 0, 0, &si, &pi))
    win_die("CreateProcess");
  WaitForSingleObject(pi.hProcess, INFINITE);

  DWORD exit_code;
  GetExitCodeProcess(pi.hProcess, &exit_code);

  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);

  if (exit_code)
    exit(1);

  DWORD n;
  if (!ReadFile(child_stdout_read, buf, 0x10000, &n, 0))
    win_die("ReadFile");
  fwrite(buf, n, 1, stdout);
  putchar('\n');
}

int main(int argc, char **argv) {
  if (argc == 1) {
    argc = 2;
    static char *default_argv[] = {
        0, "/tptp",
    };
    argv = default_argv;
  }
  for (int i = 1; i < argc; i++) {
    file = argv[i];
    if (*file == '@') {
      FILE *f = xfopen(file + 1, "r");
      for (;;) {
        fgets(buf, 0xffff, f);
        if (ferror(f)) {
          perror("fgets");
          exit(1);
        }
        if (feof(f))
          break;
        if (*buf == '\n')
          continue;
        file = xstrdup(buf);
        test();
      }
      continue;
    }
    test();
  }
  return 0;
}

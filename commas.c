#include <stdio.h>

char buf[0x10000];

int main() {
  int more = 0;
  while (1) {
    gets(buf);
    if (feof(stdin))
      break;
    if (more)
      printf(", ");
    more = 1;
    printf("%s", buf);
  }
  return 0;
}

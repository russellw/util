#include <stdio.h>

int main() {
	for (;;) {
		int c = getchar();
		if (c < 0)
			return 0;
		if (c == ';') {
			putchar('\n');
			continue;
		}
		putchar(c);
	}
}

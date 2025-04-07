#include <windows.h>

int len(char *s) {
	for (int i = 0;; i++)
		if (!s[i])
			return i;
}

void print(char *s) {
	DWORD written;
	WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), s, len(s), &written, 0);
}

int hexdigit(int n) {
	if (n < 10)
		return n + '0';
	return n - 10 + 'A';
}

void hex1(unsigned n) {
	static char buf[32];
	char *s = buf + sizeof buf - 2;
	for(int i=0;i<8;i++){
		*--s = hexdigit(n & 0xf);
		n >>= 4;
	}
	print(s);
}

void hex(unsigned long long n) {
	int *p=(int*)&n;
	hex1(p[1]);
	hex1(p[0]);
	print("\n");
}

int sum(int* a, int n) {
	int r = 0;
	int i;
	for (i = 0; i < n; i++)
		r += a[i];
	return r;
}

int f(void) {
	int a[] = {1, 2, 3, 4, 5};
	return sum(a, 5);
}

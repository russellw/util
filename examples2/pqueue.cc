#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
using std::swap;

template<class T>class PriorityQueue{
	T*p;
	unsigned cap=0;
	unsigned n=0;

 unsigned parent(unsigned i) {
	return (i + 1) / 2 - 1;
}

 unsigned left(unsigned i) {
	return (i + 1) * 2 - 1;
}

 unsigned right(unsigned i) {
	return (i + 1) * 2;
}

public:
void push(T c) {
	if (n == cap) {
		cap = cap * 2 + 1000;
		p = (T*)realloc(p, cap * sizeof(T));
	}
	p[n++] = c;
	for (auto i = n - 1; i;) {
		auto j = parent(i);
		if (!Less(p[i], p[j]))
			break;
		swap(p[i], p[j]);
		i = j;
	}
}

T pop() {
	if (!n)
		return 0;

auto	c = p[0];
	p[0] = p[--n];

	for (unsigned i = 0;;) {
		auto lo = i;

		auto j = left(i);
		if (j >= n)
			break;
		if (Less(p[j], p[i]))
			lo = j;

		j = right(i);
		if (j < n && Less(p[j], p[lo]))
			lo = j;

		if (lo == i)
			break;
		swap(p[i], p[lo]);
		i = lo;
	}
	return c;
}

unsigned size(){
	return n;
}
};

int Less(int a,int b){return a<b;}

void main(){
	PriorityQueue<int>q;
	q.push(3);
	q.push(1);
	q.push(4);
	q.push(2);
	q.push(5);
	while(q.size())
	printf("%d\n",q.pop());
}

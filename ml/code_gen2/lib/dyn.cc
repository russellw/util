#include "olivine.h"

struct List {
	size_t n;
	dyn v[9];
};

//TODO: rearrange
dyn::dyn(double a) {
	auto p = new double;
	*p = a;
	x = size_t(p) + t_num;
}

static List* list(size_t n) {
	auto r = (List*)xmalloc(offsetof(List, v) + n * sizeof(dyn));
	r->n = n;
	return r;
}

dyn list() {
	auto r = list(0);
	return dyn(r, t_list);
}

dyn list(dyn a) {
	auto r = list(1);
	r->v[0] = a;
	return dyn(r, t_list);
}

dyn list(dyn a, dyn b) {
	auto r = list(2);
	r->v[0] = a;
	r->v[1] = b;
	return dyn(r, t_list);
}

dyn list(dyn a, dyn b, dyn c) {
	auto r = list(3);
	r->v[0] = a;
	r->v[1] = b;
	r->v[2] = c;
	return dyn(r, t_list);
}

dyn list(const vector<dyn>& v) {
	auto r = list(v.size());
	memcpy(r->v, v.data(), v.size() * sizeof(dyn));
	return dyn(r, t_list);
}

dyn list(size_t op, dyn a) {
	auto r = list(2);
	r->v[0] = sym(op);
	r->v[1] = a;
	return dyn(r, t_list);
}

dyn list(size_t op, dyn a, dyn b) {
	auto r = list(3);
	r->v[0] = sym(op);
	r->v[1] = a;
	r->v[2] = b;
	return dyn(r, t_list);
}

dyn list(size_t op, dyn a, dyn b, dyn c) {
	auto r = list(4);
	r->v[0] = sym(op);
	r->v[1] = a;
	r->v[2] = b;
	r->v[3] = c;
	return dyn(r, t_list);
}

dyn list(size_t op, dyn a, dyn b, dyn c, dyn d) {
	auto r = list(5);
	r->v[0] = sym(op);
	r->v[1] = a;
	r->v[2] = b;
	r->v[3] = c;
	r->v[4] = d;
	return dyn(r, t_list);
}

dyn* dyn::begin() const {
	assert(tag() == t_list);
	auto p = (List*)(x - t_list);
	return p->v;
}

dyn* dyn::end() const {
	assert(tag() == t_list);
	auto p = (List*)(x - t_list);
	return p->v + p->n;
}

size_t dyn::kw() const {
	auto a = *this;
	if (tag() == t_list) a = a[0];
	return keyword((void*)a.x);
}

bool dyn::operator==(dyn b) const {
	if (x == b.x) return 1;
	if (tag() != b.tag()) return 0;
	switch (tag()) {
	case t_list:
	{
		auto p = (List*)(x - t_list);
		auto n = p->n;
		auto q = (List*)(b.x - t_list);
		if (n != q->n) return 0;
		for (size_t i = 0; i != n; ++i)
			if (p->v[i] != q->v[i]) return 0;
		return 1;
	}
	case t_num:
		return *(double*)(x - t_num) == *(double*)(b.x - t_num);
	}
	return 0;
}

size_t dyn::size() const {
	if (tag() != t_list) return 1;
	auto p = (List*)(x - t_list);
	return p->n;
}

dyn dyn::operator[](size_t i) const {
	assert(tag() == t_list);
	auto p = (List*)(x - t_list);
	assert(i < p->n);
	return p->v[i];
}

dyn dyn::from(size_t i) const {
	assert(tag() == t_list);
	auto p = (List*)(x - t_list);
	if (i > p->n) i = p->n;
	auto r = list(p->n - i);
	memcpy(r->v, p->v + i, (p->n - i) * sizeof(dyn));
	return dyn(r, t_list);
}

static void quote(int q, const char* s) {
	putchar(q);
	for (; *s; ++s) {
		auto c = *s;
		if (c == q || c == '\\') putchar('\\');
		putchar(c);
	}
	putchar(q);
}

void print(dyn a) {
	if (a.isSym()) {
		printf("%s", a.str());
		return;
	}
	if (a.isNum()) {
		printf("%.999g", a.num());
		return;
	}
	putchar('[');
	bool more = 0;
	for (auto b: a) {
		if (more) printf(", ");
		more = 1;
		if (b.isSym()) {
			quote('\'', b.str());
			continue;
		}
		print(b);
	}
	putchar(']');
}

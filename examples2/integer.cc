#include "ayane.h"
#include <algorithm>

const unsigned digbits = sizeof(dig) * 8;
const dig2 digbase = (dig2)1 << digbits;

static integer zero = {t_integer};

static integer* make(unsigned n) {
	auto r = (integer*)malloc(offsetof(integer, v) + n * sizeof(dig));
	r->tag = t_integer;
	r->sign = 0;
	r->n = n;
	return r;
}

static integer* copy(integer* a) {
	auto r = make(a->n);
	r->sign = a->sign;
	memcpy(r->v, a->v, a->n * sizeof(dig));
	return r;
}

static integer* norm(integer* a) {
	assert(a->tag == t_integer);
	while (a->n && !a->v[a->n - 1])
		--a->n;
	if (!a->n)
		return &zero;
	return a;
}

static void check(integer* a) {
	assert(a->tag == t_integer);
	assert(a->n >= 0);
	if (a->n)
		assert(a->v[a->n - 1]);
	else
		assert(a == &zero);
}

// Compare

bool eq(integer* a, integer* b) {
	check(a);
	check(b);

	if (a->sign ^ b->sign | a->n ^ b->n)
		return 0;
	for (unsigned i = 0; i != a->n; ++i)
		if (a->v[i] != b->v[i])
			return 0;
	return 1;
}

static bool ltu(integer* a, integer* b) {
	check(a);
	check(b);

	if (a->n != b->n)
		return a->n < b->n;
	for (auto i = a->n; i--;)
		if (a->v[i] != b->v[i])
			return a->v[i] < b->v[i];
	return 0;
}

bool lt(integer* a, integer* b) {
	check(a);
	check(b);

	if (a->sign != b->sign)
		return a->sign;
	if (a->sign)
		std::swap(a, b);
	return ltu(a, b);
}

// Arithmetic

static integer* addu(integer* a, integer* b) {
	assert(a->n >= b->n);

	auto r = make(a->n + 1);
	dig c = 0;
	for (unsigned i = 0; i != b->n; ++i) {
		auto d2 = (dig2)a->v[i] + b->v[i] + c;
		r->v[i] = d2;
		c = d2 >> digbits;
	}
	for (unsigned i = b->n; i != a->n; ++i) {
		auto d2 = (dig2)a->v[i] + c;
		r->v[i] = d2;
		c = d2 >> digbits;
	}
	r->v[a->n] = c;
	return r;
}

static integer* subu(integer* a, integer* b) {
	assert(a->n >= b->n);

	auto r = make(a->n);
	dig c = 0;
	for (unsigned i = 0; i != b->n; ++i) {
		auto d2 = (dig2)a->v[i] - b->v[i] - c;
		r->v[i] = d2;
		c = d2 >> (digbits * 2 - 1);
	}
	for (unsigned i = b->n; i != a->n; ++i) {
		auto d2 = (dig2)a->v[i] - c;
		r->v[i] = d2;
		c = d2 >> (digbits * 2 - 1);
	}
	assert(!c);
	return r;
}

integer* add(integer* a, integer* b) {
	check(a);
	check(b);

	if (ltu(a, b))
		std::swap(a, b);

	integer* r;
	if (a->sign == b->sign)
		r = addu(a, b);
	else
		r = subu(a, b);

	r->sign = a->sign;
	return norm(r);
}

integer* sub(integer* a, integer* b) {
	check(a);
	check(b);

	bool sign = 0;
	if (ltu(a, b)) {
		std::swap(a, b);
		sign = 1;
	}

	integer* r;
	if (a->sign == b->sign)
		r = subu(a, b);
	else
		r = addu(a, b);

	r->sign = a->sign ^ sign;
	return norm(r);
}

integer* mulu(integer* a, integer* b) {
	check(a);
	check(b);

	auto r = make(a->n + b->n);
	memset(r->v, 0, (a->n + b->n) * sizeof(dig));
	for (unsigned i = 0; i != a->n; ++i) {
		dig c = 0;
		for (unsigned j = 0; j != b->n; ++j) {
			auto d2 = r->v[i + j] + (dig2)a->v[i] * b->v[j] + c;
			r->v[i + j] = d2;
			c = d2 >> digbits;
		}
		r->v[i + b->n] = c;
	}
	return r;
}

integer* mul(integer* a, integer* b) {
	check(a);
	check(b);

	auto r = mulu(a, b);
	r->sign = a->sign ^ b->sign;
	return norm(r);
}

static dig divmod(integer* a, dig b) {
	dig c = 0;
	for (auto i = a->n; i--;) {
		auto d2 = (dig2)c << digbits | a->v[i];
		a->v[i] = d2 / b;
		c = d2 % b;
	}
	if (a->n && !a->v[a->n - 1]) {
		--a->n;
		assert(!(a->n && !a->v[a->n - 1]));
	}
	return c;
}

static void madd(integer* a, integer* b, dig2 d) {
	auto m = a->n - b->n - 1;
	dig c = 0;
	for (unsigned i = 0; i != b->n; ++i) {
		auto d2 = a->v[i + m] + b->v[i] * d + c;
		a->v[i + m] = d2;
		c = d2 >> digbits;
	}
	a->v[b->n + m] += c;
}

static dig msub(integer* a, integer* b, dig2 d) {
	auto m = a->n - b->n - 1;
	dig c = 0;
	for (unsigned i = 0; i != b->n; ++i) {
		auto d2 = a->v[i + m] - b->v[i] * d - c;
		a->v[i + m] = d2;
		c = d2 >> digbits;
		c = -c;
	}
	auto d2 = (dig2)a->v[b->n + m] - c;
	a->v[b->n + m] = d2;
	return d2 >> digbits;
}

static integer* divmod(integer* a, integer* b, integer*& r) {
	// Variant of Knuth volume 2
	check(a);
	check(b);
	assert(b->n);

	// b is small
	if (b->n == 1) {
		auto q = make(a->n);
		memcpy(q->v, a->v, a->n * sizeof(dig));
		r = make(1);
		*r->v = divmod(q, *b->v);
		r = norm(r);
		return norm(q);
	}

	// a < b
	if (ltu(a, b)) {
		r = copy(a);
		r->sign = 0;
		return make(0);
	}

	// Scale factor to fill top bit of b
	integer s = {t_integer, 0, 1, (dig)(digbase / ((dig2)b->v[b->n - 1] + 1))};
	assert(*s.v);

	// Scale and extend a
	r = mulu(a, &s);

	// Scale b
	b = mulu(b, &s);
	assert(!b->v[b->n - 1]);
	--b->n;
	assert(b->v[b->n - 1] >= digbase / 2);

	// Quotient
	auto q = make(r->n - b->n);
	do {
		auto d2 = ((dig2)r->v[r->n - 1] << digbits | r->v[r->n - 2]) / b->v[b->n - 1];
		dig d = d2 >> digbits ? digbase - 1 : d2;
		while (msub(r, b, d))
			madd(r, b, d--);
		q->v[r->n - b->n - 1] = d;
		--r->n;
	} while (r->n != b->n);

	// Unscale remainder
	divmod(r, *s.v);
	return norm(q);
}

integer* div(integer* a, integer* b) {
	integer* r;
	auto q = divmod(a, b, r);
	if (a->sign && r->n) {
		static integer one = {t_integer, 0, 1, 1};
		auto o = &one;
		if (!q->n)
			std::swap(q, o);
		q = addu(q, o);
	}
	q->sign = a->sign ^ b->sign;
	return norm(q);
}

integer* mod(integer* a, integer* b) {
	integer* r;
	divmod(a, b, r);
	if (a->sign && r->n)
		r = subu(b, r);
	return norm(r);
}

// Bitwise

integer* and_(integer* a, integer* b) {
	check(a);
	check(b);
	assert(!a->sign);
	assert(!b->sign);

	if (a->n < b->n)
		std::swap(a, b);

	auto r = make(a->n);
	for (unsigned i = 0; i != b->n; ++i)
		r->v[i] = a->v[i] & b->v[i];
	memset(r->v + b->n, 0, (a->n - b->n) * sizeof(dig));
	return norm(r);
}

integer* or_(integer* a, integer* b) {
	check(a);
	check(b);
	assert(!a->sign);
	assert(!b->sign);

	if (a->n < b->n)
		std::swap(a, b);

	auto r = make(a->n);
	for (unsigned i = 0; i != b->n; ++i)
		r->v[i] = a->v[i] | b->v[i];
	memcpy(r->v + b->n, a->v + b->n, (a->n - b->n) * sizeof(dig));
	return norm(r);
}

integer* xor_(integer* a, integer* b) {
	check(a);
	check(b);
	assert(!a->sign);
	assert(!b->sign);

	if (a->n < b->n)
		std::swap(a, b);

	auto r = make(a->n);
	for (unsigned i = 0; i != b->n; ++i)
		r->v[i] = a->v[i] ^ b->v[i];
	memcpy(r->v + b->n, a->v + b->n, (a->n - b->n) * sizeof(dig));
	return norm(r);
}

integer* shl(integer* a, unsigned b) {
	check(a);
	assert(!a->sign);

	auto n = b / digbits;
	auto r = make(a->n + n + 1);
	memset(r->v, 0, n * sizeof(dig));
	b %= digbits;
	if (b) {
		dig c = 0;
		for (unsigned i = 0; i != a->n; ++i) {
			r->v[n + i] = a->v[i] << b | c;
			c = a->v[i] >> (digbits - b);
		}
		r->v[n + a->n] = c;
	} else {
		--r->n;
		memcpy(r->v + n, a->v, a->n * sizeof(dig));
	}
	return norm(r);
}

integer* shr(integer* a, unsigned b) {
	check(a);
	assert(!a->sign);

	auto n = b / digbits;
	if (n >= a->n)
		return &zero;
	auto r = make(a->n - n);
	b %= digbits;
	if (b) {
		for (unsigned i = 0; i != r->n - 1; ++i)
			r->v[i] = a->v[n + i] >> b | a->v[n + i + 1] << (digbits - b);
		r->v[r->n - 1] = a->v[a->n - 1] >> b;
	} else
		memcpy(r->v, a->v + n, r->n * sizeof(dig));
	return norm(r);
}

// IO

static unsigned ceil(unsigned a, unsigned b) {
	return (a + b - 1) / b;
}

static bool isdigit(unsigned c, unsigned base) {
	assert(2 <= base && base <= 36);

	auto d = c - '0';
	if (d < 10 && d < base)
		return 1;

	if (base <= 10)
		return 0;
	base -= 10;

	d = c - 'a';
	if (d < base)
		return 1;

	d = c - 'A';
	return d < base;
}

static unsigned toi(unsigned c, unsigned base) {
	assert(isdigit(c, base));

	auto d = c - '0';
	if (d < 10)
		return d;

	d = c - 'a';
	if (d < 26)
		return d + 10;

	d = c - 'A';
	return d + 10;
}

static void madd(integer* a, dig2 b, dig c) {
	for (unsigned i = 0; i != a->n; ++i) {
		auto d2 = a->v[i] * b + c;
		a->v[i] = d2;
		c = d2 >> digbits;
	}
	if (c)
		a->v[a->n++] = c;
}

integer* make_integer(const char*& s, unsigned base) {
	assert(2 <= base && base <= 16);

	// Sign
	bool sign = 0;
	if (*s == '-') {
		++s;
		sign = 1;
	}

	// Size of string
	auto p = s;
	while (isdigit(*p, base))
		++p;
	size_t n = p - s;

	// Worst case is 2 hex digits per byte
	auto chunk = sizeof(dig) * 2;
	auto r = make(ceil(n, chunk));
	r->sign = sign;
	r->n = 0;

	// Convert digits
	for (p = s; n;) {
		dig d = 0;
		dig e = 1;
		auto i = std::min(chunk, n);
		n -= i;
		while (i--) {
			d = d * base + toi(*p++, base);
			e *= base;
		}
		madd(r, e, d);
	}

	// Results
	s = p;
	return norm(r);
}

#ifndef NDEBUG
static vec<char> buf;

void print(integer* a) {
	check(a);

	// Decimal digits per chunk
	unsigned chunk = 9;
	dig b = 1000000000;
	if (sizeof(dig) == 8) {
		chunk = 19;
		b *= b * 10;
	}

	// At least one byte for 0
	buf.resize(1 + a->n * (chunk + 1));
	auto s = buf.data() + buf.size();

	// Convert digits
	for (auto x = copy(a);;) {
		auto d = divmod(x, b);
		if (!x->n) {
			do {
				assert(s > buf.data());
				*--s = d % 10 + '0';
				d /= 10;
			} while (d);
			break;
		}
		for (auto j = chunk; j--;) {
			assert(s > buf.data());
			*--s = d % 10 + '0';
			d /= 10;
		}
	}

	// Print
	if (a->sign)
		putchar('-');
	fwrite(s, 1, buf.data() + buf.size() - s, stdout);
}
#endif

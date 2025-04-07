#include "ayane.h"

#include <set>
using std::set;

#include <gmp.h>

namespace {

// integers
struct Int: TermBase {
	mpz_t val;

	Int(): TermBase(t_int, keywords[k_int]) {
		mpz_init(val);
	}
};

bool operator<(const Int& a, const Int& b) {
	return mpz_cmp(a.val, b.val) < 0;
}

set<Int> ints;

// rationals
struct Rat: TermBase {
	mpq_t val;

	Rat(char* type): TermBase(t_rat, type) {
		mpq_init(val);
	}
};

bool operator<(const Rat& a, const Rat& b) {
	return mpq_cmp(a.val, b.val) < 0;
}

set<Rat> rats;
set<Rat> reals;

// input
vec<char> buf;

term integer(const char*& p) {
	buf.push_back(0);

	Int x;
	mpz_set_str(x.val, buf.data(), 10);

	auto r = ints.insert(x);
	if (!r.second)
		mpz_clear(x.val);
	return (term)&*r.first;
}

term rational(const char*& p, location& loc) {
	do
		buf.push_back(*p++);
	while (isdigit(*p));
	buf.push_back(0);

	Rat x(keywords[k_rat]);
	if (mpq_set_str(x.val, buf.data(), 10))
		throw input_error(loc, "bad denominator");

	mpq_canonicalize(x.val);

	auto r = rats.insert(x);
	if (!r.second)
		mpq_clear(x.val);
	return (term)&*r.first;
}

term real(const char*& p, location& loc) {
	buf.push_back(0);

	Rat x(keywords[k_real]);
	mpq_set_str(x.val, buf.data(), 10);

	if (*p == '.') {
		++p;

		buf.clear();
		do
			buf.push_back(*p++);
		while (isdigit(*p));

		mpz_t d;
		mpz_init(d);

		mpz_ui_pow_ui(d, 10, buf.size());
		mpz_mul(mpq_numref(x.val), mpq_numref(x.val), d);
		mpz_mul(mpq_denref(x.val), mpq_denref(x.val), d);

		buf.push_back(0);

		if (mpz_set_str(d, buf.data(), 10))
			throw input_error(loc, "bad decimal");
		mpz_add(mpq_numref(x.val), mpq_numref(x.val), d);

		mpz_clear(d);
	}

	if (*p == 'e' || *p == 'E') {
		++p;

		auto i = mpq_numref(x.val);
		if (*p == '-') {
			++p;
			i = mpq_denref(x.val);
		}

		size_t e;
		if (!tou(p, e))
			throw input_error(loc, "bad exponent");

		mpz_t e1;
		mpz_init(e1);

		mpz_ui_pow_ui(e1, 10, e);
		mpz_mul(i, i, e1);

		mpz_clear(e1);
	}

	mpq_canonicalize(x.val);

	auto r = reals.insert(x);
	if (!r.second)
		mpq_clear(x.val);
	return (term)&*r.first;
}

}

term number(const char*& p, location& loc) {
	assert(isdigit(p[0]) || p[0] == '-' && isdigit(p[1]));

	buf.clear();
	do
		buf.push_back(*p++);
	while (isdigit(*p));

	switch (*p) {
	default:
		return integer(p);
	case '/':
		return rational(p, loc);
	case '.':
	case 'E':
	case 'e':
		return real(p, loc);
	}
}

// output
void print_number(ostream& os, term a) {
	assert(a->tag == t_int || a->tag == t_rat);
	if (a->tag == t_int) {
		auto a1 = (Int*)a;
		os << a1->val;
	} else {
		auto a1 = (Rat*)a;
		os << a1->val;
	}
}

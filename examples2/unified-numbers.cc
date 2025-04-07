#include "ayane.h"

#include <set>
using std::set;

#include <gmp.h>

namespace {

// rationals
struct Number: TermBase {
	mpq_t val;

	Number(): TermBase(t_number, 1) {
		mpq_init(val);
	}
};

bool operator<(const Number& a, const Number& b) {
	return mpq_cmp(a.val, b.val) < 0;
}

set<Number> numbers;

// input
vec<char> buf;

void exp(const char*& p, location& loc, Number& x) {
	++p;

	auto i = mpq_numref(x.val);
	if (*p == '-') {
		++p;
		i = mpq_denref(x.val);
	}

	size_t e;
	if (!tou(p, e))
		throw syntax_error(loc, "bad exponent");

	mpz_t e1;
	mpz_init(e1);
	mpz_ui_pow_ui(e1, 10, e);
	mpz_mul(i, i, e1);
	mpz_clear(e1);
}

void real(const char*& p, location& loc, Number& x) {
	++p;

	if (!isdigit(*p))
		throw syntax_error(loc, "bad decimal");
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

	mpz_set_str(d, buf.data(), 10);
	mpz_add(mpq_numref(x.val), mpq_numref(x.val), d);

	if (*p == 'e' || *p == 'E')
		exp(p, loc, x);
}

}

term number(const char*& p, location& loc) {
	assert(isdigit(p[0]) || p[0] == '-' && isdigit(p[1]));

	buf.clear();
	do
		buf.push_back(*p++);
	while (isdigit(*p));

	Number x;
	switch (*p) {
	case '.':
		buf.push_back(0);

		mpq_set_str(x.val, buf.data(), 10);
		real(p, loc, x);
		break;
	case '/':
		do
			buf.push_back(*p++);
		while (isdigit(*p));
	default:
		buf.push_back(0);

		mpq_set_str(x.val, buf.data(), 10);
		break;
	case 'e':
	case 'E':
		buf.push_back(0);

		mpq_set_str(x.val, buf.data(), 10);
		exp(p, loc, x);
		break;
	}
	mpq_canonicalize(x.val);

	auto r = numbers.insert(x);
	if (!r.second)
		mpq_clear(x.val);
	return (term)&*r.first;
}

// output
void print_number(ostream& os, term a) {
	assert(a->tag == t_number);
	auto a1 = (Number*)a;
	os << a1->val;
}

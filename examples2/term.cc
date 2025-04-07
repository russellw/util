#include "main.h"

char*terms;

namespace {
const char *tagNames[] = {
#define _(x) #x,
#include "tags.h"
#undef _
};

struct mpz : header {
  mpz_t z;

  mpz() { mpz_init(z); }
  ~mpz() { mpz_clear(z); }
};

struct mpq : header {
  mpq_t q;

  mpq() { mpq_init(q); }
  ~mpq() { mpq_clear(q); }
};

struct compound : header {
  size_t n;
  term *data;

  compound(size_t n, term *data) : n(n), data(data) {}
  ~compound() { delete[] data; }
};
} // namespace

// other terms are boxed
term::term(tag t, const term &a) {
  size_t n = 1;
  auto data = new term[n];
  data[0] = a;
  init(t, new compound(n, data));
}

term::term(tag t, const term &a, const term &b) {
  size_t n = 2;
  auto data = new term[n];
  data[0] = a;
  data[1] = b;
  init(t, new compound(n, data));
}

term::term(tag t, const term &a, const term &b, const term &c) {
  size_t n = 3;
  auto data = new term[n];
  data[0] = a;
  data[1] = b;
  data[2] = c;
  init(t, new compound(n, data));
}

term::term(tag t, const term &a, const term &b, const term &c, const term &d) {
  size_t n = 4;
  auto data = new term[n];
  data[0] = a;
  data[1] = b;
  data[2] = c;
  data[3] = d;
  init(t, new compound(n, data));
}

term::term(tag t, const vector<term> &v) {
  auto n = v.size();
  auto data = new term[n];
  for (size_t i = 0; i < n; ++i)
    data[i] = v[i];
  init(t, new compound(n, data));
}

// fields
bool term::boxed() const {
  switch (tag(*this)) {
  case tag::False:
  case tag::True:
  case tag::Int:
  case tag::Rat:
  case tag::Real:
  case tag::Const:
  case tag::Var:
    return 0;
  }
  return 1;
}

// copy constructor and overloaded assignment operator
// TODO: move semantics?
term::term(const term &b) {
  bits = b.bits;
  if (boxed())
    ++ptr()->refs;
}

term &term::operator=(const term &b) {
  if (this == &b)
    return *this;
  bits = b.bits;
  if (boxed())
    ++ptr()->refs;
  return *this;
}

// destructor decrements reference count
term::~term() {
  if (!boxed())
    return;
  auto p = ptr();
  assert(p);
  if (!--p->refs)
    switch (tag(*this)) {
    case tag::Int:
      delete (mpz *)p;
      return;
    case tag::Rat:
    case tag::Real:
      delete (mpq *)p;
      return;
    }
  delete (compound *)p;
}

// type of this term
term::operator type() const {
  switch (tag(*this)) {
  case tag::Add:
  case tag::Sub:
  case tag::Mul:
  case tag::Div:
  case tag::Neg:
    return type((*this)[1]);
  case tag::And:
  case tag::Or:
  case tag::Not:
  case tag::Eqv:
  case tag::Eq:
  case tag::Lt:
  case tag::Le:
  case tag::All:
  case tag::Exists:
  case tag::True:
  case tag::False:
    return type(kind::Bool);
  case tag::Real:
    return type(kind::Real);
  case tag::Int:
    return type(kind::Int);
  case tag::Rat:
    return type(kind::Rat);
  case tag::DistinctObj:
    return type(kind::Individual);
  case tag::Var:
  case tag::Const:
    return type(bits & typeBits);
  case tag::Call: {
    auto ty = type((*this)[0]);
    assert(kind(ty) == kind::Fn);
    return ty[0];
  }
  }
  debug(tag(*this));
  unreachable;
}

// comparison
bool term::operator==(const term &b) const {
  if (bits == b.bits)
    return 1;
  if (tag(*this) != tag(b))
    return 0;
  switch (tag(*this)) {
  case tag::Int: {
    auto ap = (mpz *)ptr();
    auto bp = (mpz *)b.ptr();
    return !mpz_cmp(ap->z, bp->z);
  }
  case tag::Const:
  case tag::Var:
  case tag::DistinctObj:
    return 0;
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto bp = (mpq *)b.ptr();
    return mpq_equal(ap->q, bp->q);
  }
  }
  assert(boxed());
  auto ap = (compound *)ptr();
  auto bp = (compound *)b.ptr();
  auto n = ap->n;
  if (n != bp->n)
    return 0;
  auto ad = ap->data;
  auto bd = bp->data;
  for (size_t i = 0; i < n; ++i)
    if (ad[i] != bd[i])
      return 0;
  return 1;
}

int64_t term::cmp(const term &b) const {
  if (bits == b.bits)
    return 0;
  if (tag(*this) != tag(b))
    return int64_t(tag(*this)) - int64_t(tag(b));
  switch (tag(*this)) {
  case tag::Int: {
    auto ap = (mpz *)ptr();
    auto bp = (mpz *)b.ptr();
    return mpz_cmp(ap->z, bp->z);
  }
  case tag::Const:
  case tag::Var:
  case tag::DistinctObj:
    return (bits & payloadBits) - (b.bits & payloadBits);
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto bp = (mpq *)b.ptr();
    return mpq_cmp(ap->q, bp->q);
  }
  }
  assert(boxed());
  auto ap = (compound *)ptr();
  auto bp = (compound *)b.ptr();
  auto n = min(ap->n, bp->n);
  auto ad = ap->data;
  auto bd = bp->data;
  for (size_t i = 0; i < n; ++i) {
    auto c = ad[i].cmp(bd[i]);
    if (c)
      return c;
  }
  return ap->n - bp->n;
}

// arithmetic
term term::operator-() const {
  auto t = tag(*this);
  // TODO: check the generated code
  switch (t) {
  case tag::Int: {
    auto ap = (mpz *)ptr();
    auto r = new mpz;
    mpz_neg(r->z, ap->z);
    return term(t, r);
  }
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto r = new mpq;
    mpq_neg(r->q, ap->q);
    return term(t, r);
  }
  }
  unreachable;
}

term term::operator+(const term &b) const {
  auto t = tag(*this);
  assert(t == tag(b));
  switch (t) {
  case tag::Int: {
    auto ap = (mpz *)ptr();
    auto bp = (mpz *)b.ptr();
    auto r = new mpz;
    mpz_add(r->z, ap->z, bp->z);
    return term(t, r);
  }
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto bp = (mpq *)b.ptr();
    auto r = new mpq;
    mpq_add(r->q, ap->q, bp->q);
    return term(t, r);
  }
  }
  unreachable;
}

term term::operator-(const term &b) const {
  auto t = tag(*this);
  assert(t == tag(b));
  switch (t) {
  case tag::Int: {
    auto ap = (mpz *)ptr();
    auto bp = (mpz *)b.ptr();
    auto r = new mpz;
    mpz_sub(r->z, ap->z, bp->z);
    return term(t, r);
  }
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto bp = (mpq *)b.ptr();
    auto r = new mpq;
    mpq_sub(r->q, ap->q, bp->q);
    return term(t, r);
  }
  }
  unreachable;
}

term term::operator*(const term &b) const {
  auto t = tag(*this);
  assert(t == tag(b));
  switch (t) {
  case tag::Int: {
    auto ap = (mpz *)ptr();
    auto bp = (mpz *)b.ptr();
    auto r = new mpz;
    mpz_mul(r->z, ap->z, bp->z);
    return term(t, r);
  }
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto bp = (mpq *)b.ptr();
    auto r = new mpq;
    mpq_mul(r->q, ap->q, bp->q);
    return term(t, r);
  }
  }
  unreachable;
}

term term::operator/(const term &b) const {
  auto t = tag(*this);
  assert(t == tag(b));
  switch (t) {
  case tag::Int: {
    auto ap = (mpz *)ptr();
    auto bp = (mpz *)b.ptr();
    auto r = new mpz;
    // TPTP does not define integer division with unspecified rounding mode
    // but most programming languages nowadays define it as truncating
    mpz_tdiv_q(r->z, ap->z, bp->z);
    return term(t, r);
  }
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto bp = (mpq *)b.ptr();
    auto r = new mpq;
    mpq_div(r->q, ap->q, bp->q);
    return term(t, r);
  }
  }
  unreachable;
}

term term::divT(const term &b) const {
  auto t = tag(*this);
  assert(t == tag(b));
  switch (t) {
  case tag::Int: {
    auto ap = (mpz *)ptr();
    auto bp = (mpz *)b.ptr();
    auto r = new mpz;
    mpz_tdiv_q(r->z, ap->z, bp->z);
    return term(t, r);
  }
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto bp = (mpq *)b.ptr();

    mpz_t xnum_yden;
    mpz_init(xnum_yden);
    mpz_mul(xnum_yden, mpq_numref(ap->q), mpq_denref(bp->q));

    mpz_t xden_ynum;
    mpz_init(xden_ynum);
    mpz_mul(xden_ynum, mpq_denref(ap->q), mpq_numref(bp->q));

    auto r = new mpq;
    mpz_tdiv_q(mpq_numref(r->q), xnum_yden, xden_ynum);

    mpz_clear(xden_ynum);
    mpz_clear(xnum_yden);
    return term(t, r);
  }
  }
  unreachable;
}

term term::divF(const term &b) const {
  auto t = tag(*this);
  assert(t == tag(b));
  switch (t) {
  case tag::Int: {
    auto ap = (mpz *)ptr();
    auto bp = (mpz *)b.ptr();
    auto r = new mpz;
    mpz_fdiv_q(r->z, ap->z, bp->z);
    return term(t, r);
  }
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto bp = (mpq *)b.ptr();

    mpz_t xnum_yden;
    mpz_init(xnum_yden);
    mpz_mul(xnum_yden, mpq_numref(ap->q), mpq_denref(bp->q));

    mpz_t xden_ynum;
    mpz_init(xden_ynum);
    mpz_mul(xden_ynum, mpq_denref(ap->q), mpq_numref(bp->q));

    auto r = new mpq;
    mpz_fdiv_q(mpq_numref(r->q), xnum_yden, xden_ynum);

    mpz_clear(xden_ynum);
    mpz_clear(xnum_yden);
    return term(t, r);
  }
  }
  unreachable;
}

term term::divE(const term &b) const {
  auto t = tag(*this);
  assert(t == tag(b));
  switch (t) {
  case tag::Int: {
    auto ap = (mpz *)ptr();
    auto bp = (mpz *)b.ptr();
    auto r = new mpz;
    mpz_ediv_q(r->z, ap->z, bp->z);
    return term(t, r);
  }
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto bp = (mpq *)b.ptr();

    mpz_t xnum_yden;
    mpz_init(xnum_yden);
    mpz_mul(xnum_yden, mpq_numref(ap->q), mpq_denref(bp->q));

    mpz_t xden_ynum;
    mpz_init(xden_ynum);
    mpz_mul(xden_ynum, mpq_denref(ap->q), mpq_numref(bp->q));

    auto r = new mpq;
    mpz_ediv_q(mpq_numref(r->q), xnum_yden, xden_ynum);

    mpz_clear(xden_ynum);
    mpz_clear(xnum_yden);
    return term(t, r);
  }
  }
  unreachable;
}

term term::remT(const term &b) const {
  auto t = tag(*this);
  assert(t == tag(b));
  switch (t) {
  case tag::Int: {
    auto ap = (mpz *)ptr();
    auto bp = (mpz *)b.ptr();
    auto r = new mpz;
    mpz_tdiv_r(r->z, ap->z, bp->z);
    return term(t, r);
  }
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto bp = (mpq *)b.ptr();

    mpz_t xnum_yden;
    mpz_init(xnum_yden);
    mpz_mul(xnum_yden, mpq_numref(ap->q), mpq_denref(bp->q));

    mpz_t xden_ynum;
    mpz_init(xden_ynum);
    mpz_mul(xden_ynum, mpq_denref(ap->q), mpq_numref(bp->q));

    auto r = new mpq;
    mpz_tdiv_r(mpq_numref(r->q), xnum_yden, xden_ynum);

    mpz_clear(xden_ynum);
    mpz_clear(xnum_yden);
    return term(t, r);
  }
  }
  unreachable;
}

term term::remF(const term &b) const {
  auto t = tag(*this);
  assert(t == tag(b));
  switch (t) {
  case tag::Int: {
    auto ap = (mpz *)ptr();
    auto bp = (mpz *)b.ptr();
    auto r = new mpz;
    mpz_fdiv_r(r->z, ap->z, bp->z);
    return term(t, r);
  }
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto bp = (mpq *)b.ptr();

    mpz_t xnum_yden;
    mpz_init(xnum_yden);
    mpz_mul(xnum_yden, mpq_numref(ap->q), mpq_denref(bp->q));

    mpz_t xden_ynum;
    mpz_init(xden_ynum);
    mpz_mul(xden_ynum, mpq_denref(ap->q), mpq_numref(bp->q));

    auto r = new mpq;
    mpz_fdiv_r(mpq_numref(r->q), xnum_yden, xden_ynum);

    mpz_clear(xden_ynum);
    mpz_clear(xnum_yden);
    return term(t, r);
  }
  }
  unreachable;
}

term term::remE(const term &b) const {
  auto t = tag(*this);
  assert(t == tag(b));
  switch (t) {
  case tag::Int: {
    auto ap = (mpz *)ptr();
    auto bp = (mpz *)b.ptr();
    auto r = new mpz;
    mpz_ediv_r(r->z, ap->z, bp->z);
    return term(t, r);
  }
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto bp = (mpq *)b.ptr();

    mpz_t xnum_yden;
    mpz_init(xnum_yden);
    mpz_mul(xnum_yden, mpq_numref(ap->q), mpq_denref(bp->q));

    mpz_t xden_ynum;
    mpz_init(xden_ynum);
    mpz_mul(xden_ynum, mpq_denref(ap->q), mpq_numref(bp->q));

    auto r = new mpq;
    mpz_ediv_r(mpq_numref(r->q), xnum_yden, xden_ynum);

    mpz_clear(xden_ynum);
    mpz_clear(xnum_yden);
    return term(t, r);
  }
  }
  unreachable;
}

term term::ceil() const {
  auto t = tag(*this);
  switch (t) {
  case tag::Int:
    return *this;
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto r = new mpq;
    mpz_cdiv_q(mpq_numref(r->q), mpq_numref(ap->q), mpq_denref(ap->q));
    return term(t, r);
  }
  }
  unreachable;
}

term term::floor() const {
  auto t = tag(*this);
  switch (t) {
  case tag::Int:
    return *this;
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto r = new mpq;
    mpz_fdiv_q(mpq_numref(r->q), mpq_numref(ap->q), mpq_denref(ap->q));
    return term(t, r);
  }
  }
  unreachable;
}

term term::trunc() const {
  auto t = tag(*this);
  switch (t) {
  case tag::Int:
    return *this;
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto r = new mpq;
    mpz_tdiv_q(mpq_numref(r->q), mpq_numref(ap->q), mpq_denref(ap->q));
    return term(t, r);
  }
  }
  unreachable;
}

term term::round() const {
  auto t = tag(*this);
  switch (t) {
  case tag::Int:
    return *this;
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto r = new mpq;
    mpz_round(mpq_numref(r->q), mpq_numref(ap->q), mpq_denref(ap->q));
    return term(t, r);
  }
  }
  unreachable;
}

// conversion
bool term::isInt() const {
  auto t = tag(*this);
  switch (t) {
  case tag::Int:
    return 1;
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    return !mpz_cmp_ui(mpq_denref(ap->q), 1);
  }
  }
  unreachable;
}

term term::toInt() const {
  auto t = tag(*this);
  switch (t) {
  case tag::Int:
    return *this;
  case tag::Rat:
  case tag::Real: {
    auto ap = (mpq *)ptr();
    auto r = new mpz;
    // different languages have different conventions
    // on the default rounding mode for converting fractions to integers
    // TPTP defines it as floor, so that is used here
    // To use a different rounding mode
    // explicity round the rational number first
    // and then convert to integer
    mpz_fdiv_q(r->z, mpq_numref(ap->q), mpq_denref(ap->q));
    return term(tag::Int, r);
  }
  }
  unreachable;
}

term term::toRat() const {
  switch (tag(*this)) {
  case tag::Int: {
    auto ap = (mpz *)ptr();
    auto r = new mpq;
    mpz_set(mpq_numref(r->q), ap->z);
    return term(tag::Rat, r);
  }
  case tag::Rat:
  case tag::Real:
    return term(tag::Rat, ptr());
  }
  unreachable;
}

term term::toReal() const {
  switch (tag(*this)) {
  case tag::Int: {
    auto ap = (mpz *)ptr();
    auto r = new mpq;
    mpz_set(mpq_numref(r->q), ap->z);
    return term(tag::Real, r);
  }
  case tag::Rat:
  case tag::Real:
    return term(tag::Real, ptr());
  }
  unreachable;
}

// compound terms
size_t term::size() const {
  if (!boxed())
    return 0;
  return ((compound *)ptr())->n;
}

term term::operator[](size_t i) const {
  assert(boxed());
  auto p = (compound *)ptr();
  assert(i < p->n);
  return p->data[i];
}

// make numbers
term int1(int n) {
  auto r = new mpz;
  mpz_set_si(r->z, n);
  return term(tag::Int, r);
}

term int1(const char *s) {
  auto r = new mpz;
  if (mpz_set_str(r->z, s, 10))
    throw "invalid integer";
  return term(tag::Int, r);
}

term rat(int n, unsigned d) {
  auto r = new mpq;
  mpq_set_si(r->q, n, d);
  mpq_canonicalize(r->q);
  return term(tag::Rat, r);
}

term rat(const char *s) {
  auto r = new mpq;
  if (mpq_set_str(r->q, s, 10))
    throw "invalid rational";
  mpq_canonicalize(r->q);
  return term(tag::Rat, r);
}

term real(int n, unsigned d) {
  auto r = new mpq;
  mpq_set_si(r->q, n, d);
  mpq_canonicalize(r->q);
  return term(tag::Real, r);
}

term real(const char *s) {
  // sign
  assert(*s != '+');
  bool sign = 0;
  if (*s == '-') {
    ++s;
    sign = 1;
  }

  // integer part
  mpz_t mpz;
  mpz_init(mpz);
  auto t = s;
  if (isDigit(*t)) {
    do
      ++t;
    while (isDigit(*t));
    bufCpy(s, t);
    if (mpz_set_str(mpz, buf, 10))
      throw "invalid integer part";
    s = t;
  }

  // decimal part
  mpz_t mantissa;
  mpz_init(mantissa);
  size_t scale = 0;
  if (*s == '.') {
    ++s;
    t = s;
    if (isDigit(*t)) {
      do
        ++t;
      while (isDigit(*t));
      bufCpy(s, t);
      if (mpz_set_str(mantissa, buf, 10))
        throw "invalid decimal part";
      scale = t - s;
      s = t;
    }
  }
  mpz_t powScale;
  mpz_init(powScale);
  mpz_ui_pow_ui(powScale, 10, scale);

  // mantissa += mpz * 10^scale
  mpz_addmul(mantissa, mpz, powScale);

  // sign
  if (sign)
    mpz_neg(mantissa, mantissa);

  // result = scaled mantissa
  auto r = new mpq;
  mpq_set_num(r->q, mantissa);
  mpq_set_den(r->q, powScale);

  // exponent
  bool exponentSign = 0;
  auto exponent = 0UL;
  if (*s == 'e' || *s == 'e') {
    ++s;
    switch (*s) {
    case '-':
      exponentSign = 1;
    case '+':
      ++s;
      break;
    }
    errno = 0;
    exponent = strtoul(s, 0, 10);
    if (errno)
      throw strerror(errno);
  }
  mpz_t powExponent;
  mpz_init(powExponent);
  mpz_ui_pow_ui(powExponent, 10, exponent);
  if (exponentSign)
    mpz_mul(mpq_denref(r->q), mpq_denref(r->q), powExponent);
  else
    mpz_mul(mpq_numref(r->q), mpq_numref(r->q), powExponent);

  // reduce result to lowest terms
  mpq_canonicalize(r->q);

  // cleanup
  mpz_clear(powExponent);
  mpz_clear(powScale);
  mpz_clear(mantissa);
  mpz_clear(mpz);

  // wrap result in term designating it as a real number
  return term(tag::Real, r);
}

// print
void print(tag t) { print(tagNames[(int)t]); }

void print(const term &a) {
  switch (tag(a)) {
  case tag::Var: {
    if (a.flag())
      putchar('`');
    auto i = a.idx();
    if (i < 26)
      putchar('A' + i);
    else
      printf("Z%zu", i - 25);
    return;
  }
  case tag::Const: {
    auto i = a.idx();
    if (a.flag()) {
      printf("_%zu", i);
      return;
    }
    print(getSym(i));
    return;
  }
  case tag::Int: {
    auto p = (mpz *)a.ptr();
    mpz_out_str(stdout, 10, p->z);
    return;
  }
  case tag::Rat: {
    auto p = (mpq *)a.ptr();
    mpq_out_str(stdout, 10, p->q);
    if (a.isInt())
      printf("/1");
    return;
  }
  case tag::Real: {
    auto p = (mpq *)a.ptr();
    printf("%f", mpq_get_d(p->q));
    return;
  }
  }
  print(tag(a));
  if (!a.size())
    return;
  putchar('(');
  for (size_t i = 0; i < a.size(); ++i) {
    if (i)
      printf(", ");
    print(a[i]);
  }
  putchar(')');
}

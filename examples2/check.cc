#include "main.h"

[[noreturn]] static void err(size_t expected,size_t actual){
      sprintf(buf, "expected %zu args, received %zu", expected,actual);
      throw buf;
}

void check(term a, type ty) {
  // in first-order logic, a function cannot return a function
  // that would be higher-order logic
  // the code should be written so that
  // neither the top-level callers nor the recursive calls
  // can ever ask for a function to be returned
  assert(kind(ty) != kind::Fn);

  // all symbols used in a formula must have specified types
  // by the time this check is run
  // otherwise, there would be no way of knowing
  // whether the types they will be given in the future
  // would have passed the check
  if (ty == kind::Unknown)
    throw "unspecified type";

  // need to handle calls before checking the type of this term
  // because the type of a call is only well-defined
  // if the type of the function is well-defined
  if (tag(a) == tag::Call) {
    auto fty = type(a[0]);
    if (kind(fty) != kind::Fn)
      throw "called a non-function";
    if (fty.size() != a.size())
      err( fty.size() - 1,
              a.size() - 1);
    if (fty[0] != ty)
      throw "type mismatch";
    for (size_t i = 1; i < a.size(); ++i) {
      switch (kind(fty[i])) {
      case kind::Bool:
      case kind::Fn:
        throw "invalid type for function argument";
      }
      check(a[i], fty[i]);
    }
    return;
  }

  if (type(a) != ty)
    throw "type mismatch";

  switch (tag(a)) {
  case tag::Real:
  case tag::Rational:
  case tag::DistinctObj:
  case tag::Integer:
    return;
  case tag::Var:
    switch (kind(ty)) {
    case kind::Bool:
    case kind::Fn:
      throw "invalid type for variable";
    }
    return;
  case tag::Const:
  case tag::Sym:
    // in first-order logic, functions can only appear when called
    // not as first-class values
    if (kind(ty) == kind::Fn)
      throw "uncalled function";
    return;
  case tag::And:
  case tag::Or:
  case tag::Not:
  case tag::Eqv:
    for (size_t i = 0; i < a.size(); ++i)
      check(a[i], kind::Bool);
    return;
  case tag::All:
  case tag::Exists:
    check(a[0], kind::Bool);
    return;
  case tag::Eq: {
    ty = type(a[0]);
    switch (kind(ty)) {
    case kind::Bool:
    case kind::Fn:
      throw "invalid type for equality";
    }
    check(a[0], ty);
    check(a[1], ty);
    return;
  }
  case tag::Lt:
  case tag::Le: {
  	if(a.size()!=2)
  		err(2,a.size());
    ty = type(a[0]);
    switch (kind(ty)) {
    case kind::Integer:
    case kind::Rational:
    case kind::Real:
      break;
    default:
      throw "invalid type for comparison";
    }
    check(a[0], ty);
    check(a[1], ty);
    return;
  }
  case tag::Add:
  case tag::Sub:
  case tag::Mul:
  case tag::DivE:
  case tag::DivF:
  case tag::DivT:
  case tag::RemE:
  case tag::RemF:
  case tag::RemT:
  	{
  	if(a.size()!=2)
  		err(2,a.size());
    ty = type(a[0]);
    switch (kind(ty)) {
    case kind::Integer:
    case kind::Rational:
    case kind::Real:
      break;
    default:
      throw "invalid type for arithmetic";
    }
    for (size_t i = 0; i < a.size(); ++i)
      check(a[i], ty);
    return;
  }
  case tag::IsInteger:
  case tag::IsRational:
  case tag::ToReal:
  case tag::ToInteger:
  case tag::ToRational:
  case tag::Neg:
  case tag::Round:
  case tag::Trunc:
  case tag::Ceil:
  case tag::Floor:
  	{
  	if(a.size()!=1)
  		err(1,a.size());
    ty = type(a[0]);
    switch (kind(ty)) {
    case kind::Integer:
    case kind::Rational:
    case kind::Real:
      break;
    default:
      throw "invalid type for arithmetic";
    }
    for (size_t i = 0; i < a.size(); ++i)
      check(a[i], ty);
    return;
  }
  case tag::Div:
  	{
  	if(a.size()!=2)
  		err(2,a.size());
    ty = type(a[0]);
    switch (kind(ty)) {
    case kind::Rational:
    case kind::Real:
      break;
    default:
      throw "invalid type for rational division";
    }
    for (size_t i = 0; i < a.size(); ++i)
      check(a[i], ty);
    return;
  }
  }
  debug(tag(a));
  unreachable;
}

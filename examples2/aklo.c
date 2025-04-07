#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#ifndef O_BINARY
#define O_BINARY 0
#endif

#ifdef NDEBUG
#define debug(fmt, x)
#else
#define debug(fmt, x)                                                          \
  printf("%s:%d:%s: %" #fmt "\n", __FILE__, __LINE__, #x, x);
#endif

enum {
  t_list,
  t_number,
  t_string,
};

int tag(void *x) { return *((char *)x); }

// list

struct list {
  char tag;
  void *car;
  void *cdr;
};

struct list *cons(void *x, void *y) {
  struct list *r = malloc(sizeof(struct list));
  r->tag = t_list;
  r->car = x;
  r->cdr = y;
  return r;
}

void *car(void *x) {
  assert(tag(x) == t_list);
  return ((struct list *)x)->car;
}

void *cdr(void *x) {
  assert(tag(x) == t_list);
  return ((struct list *)x)->cdr;
}

struct list nil = {t_list, &nil, &nil};

struct list *list2(void *x, void *y) {
  return cons(x, cons(y, &nil));
}

// number

struct number {
  char tag;
  double val;
};

struct number *make_number(double val) {
  struct number *r = malloc(sizeof(struct number));
  r->tag = t_number;
  r->val = val;
  return r;
}

double number_double(void *x) {
  assert(tag(x) == t_number);
  return ((struct number *)x)->val;
}

size_t number_size_t(void *x) {
  assert(tag(x) == t_number);
  return (size_t)(((struct number *)x)->val);
}

// string

size_t hash(size_t len, char *s) {
  size_t h = 2166136261u;
  while (len--) {
    h ^= *s++;
    h *= 16777619;
  }
  return h;
}

struct string {
  char tag;
  unsigned len;
  char s[8];
};

size_t string_len(void *x) {
  assert(tag(x) == t_string);
  return ((struct string *)x)->len;
}

char *string_s(void *x) {
  assert(tag(x) == t_string);
  return ((struct string *)x)->s;
}

enum {
#define _(k, s) k,
#include "keywords.h"
#undef _
  nkeywords
};

char *keywords1[] = {
#define _(k, s) s,
#include "keywords.h"
#undef _
};

size_t nslots = 256;
size_t nstrings = nkeywords;
struct string **strings;
struct string keywords[nkeywords];

size_t slot(size_t nslots, struct string **strings, size_t len, char *s) {
  size_t mask = nslots - 1;
  for (size_t i = hash(len, s) & mask;;) {
    // not present - return new slot
    if (!strings[i])
      return i;

    // present - return existing slot
    if (strings[i]->len == len && memcmp(strings[i]->s, s, len) == 0)
      return i;

    i = (i + 1) & mask;
  }
}

void init_strings(void) {
  strings = calloc(nslots, sizeof(void *));
  for (size_t k = 0; k != nkeywords; k++) {
    char *s = keywords1[k];
    size_t len = strlen(s);
    assert(len < sizeof(keywords->s));
    size_t i = slot(nslots, strings, len, s);
    assert(!strings[i]);

    struct string *p = &keywords[k];
    p->tag = t_string;
    p->len = len;
    memcpy(p->s, s, len);
    strings[i] = p;
  }
}

struct string *intern(size_t len, char *s) {
  size_t i = slot(nslots, strings, len, s);
  if (strings[i])
    return strings[i];

  if (++nstrings > nslots * 3 / 4) {
    size_t nslots1 = nslots * 2;
    struct string **strings1 = calloc(nslots1, sizeof(void *));
    for (size_t i = 0; i != nslots; ++i) {
      struct string *p = strings[i];
      if (p)
        strings1[slot(nslots1, strings1, p->len, p->s)] = p;
    }
    free(strings);
    nslots = nslots1;
    strings = strings1;
    i = slot(nslots, strings, len, s);
  }

  struct string *p = malloc(offsetof(struct string, s) + len + 1);
  p->tag = t_string;
  p->len = len;
  memcpy(p->s, s, len);
  p->s[len] = 0;
  strings[i] = p;
  return p;
}

size_t keyword(void *x) {
  size_t i = (char *)x - (char *)keywords;
  return i / sizeof *keywords;
}

// string buffer

char buf[256];
char *bufp;

void buf_putc(int c) {
  if (bufp == buf + sizeof buf) {
    fprintf(stderr, "String buffer overflow\n");
    exit(1);
  }
  *bufp++ = c;
}

// input

char *read_file(char *file) {
  int f = open(file, O_RDONLY | O_BINARY);
  struct stat st;
  if (f < 0 || fstat(f, &st)) {
    perror(file);
    exit(1);
  }
  char *r = malloc(st.st_size + 1);
  if (read(f, r, st.st_size) != st.st_size) {
    perror(file);
    exit(1);
  }
  r[st.st_size] = 0;
  close(f);
  return r;
}

int delimiter(int c) {
  if (c <= ' ')
    return 1;
  switch (c) {
  case '"':
  case '#':
  case '(':
  case ')':
  case ',':
  case '.':
  case ':':
  case ';':
  case '[':
  case '\'':
  case ']':
  case '`':
  case '{':
  case '}':
    return 1;
  }
  return 0;
}

char *file;
unsigned line;

#ifdef _MSC_VER
__declspec(noreturn)
#endif
    void err(char *fmt, ...) {
  va_list va;
  fprintf(stderr, "%s:%u: ", file, line);
  va_start(va, fmt);
  vfprintf(stderr, fmt, va);
  va_end(va);
  fputc('\n', stderr);
  exit(1);
}

char *ptr;

int tok;
void *tokval;

void lex(void) {
  char *ptr1;
  for (;;)
    switch (tok = *ptr) {
    // whitespace
    case '\n':
      ++line;
    // fallthru
    case ' ':
    case '\f':
    case '\r':
    case '\t':
    case '\v':
      ++ptr;
      continue;

    // comment
    case ';':
      ptr = strchr(ptr, '\n');
      if (!ptr)
        ptr = "";
      continue;

    // string
    case '"':
    case '\'':
      bufp = buf;
      unsigned line1 = line;
      for (ptr1 = ptr + 1; *ptr1 != tok;) {
        unsigned c = *ptr1++;
        switch (c) {
        case '\n':
          line++;
          break;
        case '\r':
          if (*ptr1 == '\n') {
            line++;
            c = *ptr1++;
          }
          break;

        case '\\':
          switch (c = *ptr1++) {
          case 'a':
            c = '\a';
            break;
          case 'b':
            c = '\b';
            break;
          case 'e':
            c = 27;
            break;
          case 'f':
            c = '\f';
            break;
          case 'n':
            c = '\n';
            break;
          case 'r':
            c = '\r';
            break;
          case 't':
            c = '\t';
            break;
          case 'v':
            c = '\v';
            break;
          }
          break;

        case 0:
          line = line1;
          err("unclosed quote");
        }
        buf_putc(c);
      }
      ptr = ptr1 + 1;
      tokval = intern(bufp - buf, buf);
      return;

    // number
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      errno = 0;
      double val = strtod(ptr, &ptr);
      if (errno)
        err(strerror(errno));
      if (!delimiter(*ptr))
        err("unknown number format");
      tok = '\'';
      tokval = make_number(val);
      return;

    default:
      // punctuation
      if (delimiter(*ptr)) {
        ++ptr;
        return;
      }

      // string
      ptr1 = ptr;
      do
        ++ptr1;
      while (!delimiter(*ptr1));
      tok = '\'';
      tokval = intern(ptr1 - ptr, ptr);
      ptr = ptr1;
      return;

    // end of file
    case 0:
      return;
    }
}

struct list *parse_list(unsigned line1, int ending);

void *parse_term(void) {
  unsigned line1 = line;
  void *tokval1 = tokval;
  switch (tok) {
  case '"':
    lex();
    return list2(keywords + k_quote, tokval1);
  case '(':
    lex();
    return parse_list(line1, ')');
  case '[':
    lex();
    return parse_list(line1, ']');
  case '\'':
    lex();
    return tokval1;
  }
  if (isprint(tok))
    err("'%c': unknown character", tok);
  else
    err("'\\x%02x': unknown character", tok);
}

struct list *parse_list(unsigned line1, int ending) {
  if (tok == ending) {
    lex();
    return &nil;
  }
  switch (tok) {
  case ')':
  case ']':
    err("unexpected '%c'", tok);
  case 0:
    line = line1;
    err("unterminated list");
  }
  void *x = parse_term();
  return cons(x, parse_list(line1, ending));
}

struct list *parse(char *s) {
  line = 1;
  ptr = s;
  lex();
  return parse_list(line, 0);
}

// output

void print(void *x) {
  switch (tag(x)) {
  case t_list:
    putchar('(');
    while (x != &nil) {
      print(car(x));
      x = cdr(x);
      if (x != &nil)
        putchar(' ');
    }
    putchar(')');
    break;
  case t_number:
    printf("%.20g", number_double(x));
    break;
  case t_string:
    fwrite(string_s(x), string_len(x), 1, stdout);
    break;
  default:
    assert(0);
  }
}

// main

void help(void) {
  printf("Usage: aklo [options] file\n"
         "\n"
         "-h  Show available options\n"
         "-v  Show version\n");
}

int main(int argc, char **argv) {
  for (int i = 1; i != argc; i++) {
    char *s = argv[i];
    if (*s != '-') {
      if (file) {
        help();
        return 1;
      }
      file = s;
      continue;
    }
    while (*s == '-')
      ++s;
    switch (*s) {
    case '?':
    case 'h':
      help();
      return 0;
    case 'V':
    case 'v':
      puts("aklo 0");
      return 0;
    default:
      fprintf(stderr, "%s: unknown option\n", argv[i]);
      return 1;
    }
  }
  if (!file) {
    help();
    return 1;
  }

  init_strings();

  char *s = read_file(file);
  print(parse(s));
  free(s);
  return 0;
}

from etc import *

__all__ = [
    "parse",
]

# tokenizer
filename = ""
text = ""
ti = 0
line = 0
tok = None


def constituent(c):
    if c.isspace():
        return
    if c in "()[];{}":
        return
    return 1


def lex():
    global ti
    global tok
    global line
    while ti < len(text):
        i = ti

        # whitespace
        if text[ti].isspace():
            if text[ti] == "\n":
                line += 1
            ti += 1
            continue

        # comment
        if text[ti] == ";":
            while text[ti] != "\n":
                ti += 1
            continue
        if text[ti] == "{":
            lin = line
            while text[ti] != "}":
                if ti == len(text):
                    raise Exception("%s:%d: unclosed block comment" % (filename, lin))
                if text[ti] == "\n":
                    line += 1
                ti += 1
            ti += 1
            continue

        # number
        if text[ti].isdigit() or (text[ti] == "-" and text[ti + 1].isdigit()):
            ti += 1
            while text[ti].isalnum():
                ti += 1
            if text[ti] == ".":
                ti += 1
                while text[ti].isalnum():
                    ti += 1
            tok = text[i:ti]
            return

        # word
        if constituent(text[ti]):
            while constituent(text[ti]):
                ti += 1
            tok = text[i:ti]
            return

        # punctuation
        if text[ti] in "()":
            ti += 1
            tok = text[i:ti]
            return

        # none of the above
        raise Exception("%s:%d: stray '%c' in program" % (filename, line, text[ti]))
    tok = None


# parser
def eat(k):
    if tok == k:
        lex()
        return 1


def assoc(a):
    if len(a) <= 3:
        return a
    o = a[0]
    return (o, a[1], assoc(((o,) + a[2:])))


def pairwise(a):
    if len(a) <= 3:
        return a
    o = a[0]

    u = ["do"]
    b = [0]
    for i in range(1, len(a)):
        b.append(gensym())
        u.append(("=", b[i], a[i]))

    v = ["and"]
    for i in range(1, len(a) - 1):
        v.append((o, b[i], b[i + 1]))
    u.append(tuple(v))

    return tuple(u)


def expr():
    line1 = line
    if tok == "(":
        v = []
        lex()
        while not eat(")"):
            v.append(expr())
        a = tuple(v)

        if not a:
            return a
        if a[0] in ("+", "*", "and", "or"):
            return assoc(a)
        if a[0] in ("==", "/=", "<", "<=", ">", ">="):
            return pairwise(a)
        if a[0] == "assert":
            return (
                "if",
                a[1],
                0,
                ("err", ("quote", ("%s:%d: assert failed" % (filename, line1)))),
            )
        if a[0] == "when":
            return "if", a[1], (("do",) + a[1:]), 0

        return a
    if tok[0].isdigit() or (tok[0] == "-" and len(tok) > 1 and tok[1].isdigit()):
        a = int(tok)
        lex()
        return a
    s = tok
    lex()
    return s


def parse(filename1):
    global text
    global filename
    global line
    global ti
    filename = filename1
    text = open(filename).read() + "\n"
    ti = 0
    line = 1
    lex()
    v = []
    while tok:
        v.append(expr())
    return v

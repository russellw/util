# Uppercase first letter of comments
# Internal tool, designed for this project only
# Does NOT work for arbitrary Python code
import re

import etc
import py


def f(s):
    m = re.match("( *# )([a-z])(.*)", s)
    if m:
        return f"{m[1]}{m[2].upper()}{m[3]}"
    return s


for file in py.srcFiles():
    v = etc.readLines(file)
    for i in range(len(v)):
        if py.comment(v[i]) and not (i and py.comment(v[i - 1])):
            v[i] = f(v[i])
    etc.writeLines(file, v)

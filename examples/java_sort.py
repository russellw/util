import re

import common


# SORT
def block(v, dent, i):
    # end
    if end(v, dent, i):
        return

    # comments are considered part of the following block
    while re.match(r"\s*//", v[i]):
        i += 1

    # so are annotations
    while re.match(r"\s*@", v[i]):
        i += 1

    # braced block
    if v[i].endswith("{"):
        while not (common.indent(v, i) == dent and re.match(r"\s*},?$", v[i])):
            i += 1

    # if there was no brace, just one line
    return i + 1


def end(v, dent, i):
    if common.indent(v, i) < dent:
        return 1
    if re.match(r"\s*//$", v[i]):
        return 1
    if re.match(r"\s*// SORT$", v[i]):
        return 1


def f(v):
    i = 0
    while i < len(v):
        if not re.match(r"\s*// SORT$", v[i]):
            i += 1
            continue

        dent = common.indent(v, i)
        i += 1

        j = i
        r = []
        while 1:
            while j < len(v) and not v[j]:
                j += 1
            k = block(v, dent, j)
            if not k:
                break
            r.append(v[j:k])
            j = k
        assert r

        r.sort(key=key)
        v[i:j] = common.cat(r)
        i = j


def key(v):
    i = 0

    # skip comments
    while re.match(r"\s*//", v[i]):
        i += 1

    # skip annotations
    while re.match(r"\s*@", v[i]):
        i += 1
    s = v[i]

    # composite key
    r = []

    # public first
    if re.match(r"\s*public ", s):
        r.append(0)
    else:
        r.append(1)

    # name
    while 1:
        m = re.search(r"(\w+) = ", s)
        if m:
            r.append(m[1])
            break

        m = re.search(r"(\w+)\(", s)
        if m:
            r.append(m[1])
            break

        m = re.search(r"(\w+);", s)
        if m:
            r.append(m[1])
            break

        r.append("")
        break

    # raw text as tiebreaker
    r.append(s)
    return r


common.modify_files(f, common.args_java_files())

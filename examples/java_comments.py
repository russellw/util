import re

import common


def f(v):
    for i in range(len(v)):
        s = v[i]
        if not re.match(r"\s*//", s):
            continue

        # IDEA does actually need noinspection to stay lowercase with no leading space
        if re.match(r"\s*//noinspection ", s):
            continue

        m = re.match(r"(\s*)//\s*sort$", s, re.IGNORECASE)
        if m:
            v[i] = f"{m[1]}// SORT"
            continue

        m = re.match(r"(\s*)//\s*todo:\s*(.*)", s, re.IGNORECASE)
        if m:
            # do this indirectly to make sure
            # this doesn't show up in a search for to-do items
            s = "todo".upper()
            v[i] = f"{m[1]}// {s}: {m[2]}"
            continue

        m = re.match(r"(\s*)//\s*(https?:.*)", s)
        if m:
            v[i] = f"{m[1]}// {m[2]}"
            continue

        m = re.match(r"(\s*)//\s*(.*)", v[i])
        if m:
            v[i] = f"{m[1]}// {m[2]}"


common.modify_files(f, common.args_java_files())

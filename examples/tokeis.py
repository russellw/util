import os
import subprocess

packages = {}
totlc = {}
totpc = {}
ls = set()
ps = []

for p in os.listdir("."):
    ps.append(p)
    cmd = "tokei", p
    s = subprocess.check_output(cmd, encoding="utf-8")
    v = s.splitlines()
    v = v[3:-3]
    packages[p] = {}
    n = 0
    for s in v:
        l = s[:18].strip()
        c = int(s[39:52])
        if c:
            n += c
            packages[p][l] = c
            totlc[l] = totlc.get(l, 0) + c
            ls.add(l)
    totpc[p] = n

ls = list(ls)
ls.sort(reverse=True, key=lambda l: totlc[l])

ps.sort(key=lambda p: totpc[p])

print("name", end="\t")
for l in ls:
    print(l, end="\t")
print("total")

for p in ps:
    print(p, end="\t")
    n = 0
    for l in ls:
        c = packages[p].get(l, 0)
        print(c, end="\t")
        n += c
    print(n)
